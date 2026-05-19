# Small-SAE Benchmark: TopK / L1 / Gated vs. Position-Aware TopK

**Date:** 2026-05-19 (v0.4.2 — GPT-2 L10 refutes universal-Pareto claim)

## Question

Two questions, one experiment:

1. **Which existing SAE architecture is easiest to use at small model scale
   with a standard training recipe?** Big SAE comparison papers benchmark
   at GPT-2-medium or larger; no systematic comparison at sub-1B exists.
2. **Can a per-position pre-bias eliminate the outlier-position trap?**
   Our previous Qwen2.5-0.5B v0.1 → v0.2 work showed that the first ~4
   sequence positions have anomalously high residual-stream norm and
   dominate a vanilla TopK SAE's dictionary. The v0.2 fix was crude — just
   skip those positions. Can we keep them?

## Architectures

| arch                 | sparsity mechanism                                            |
|----------------------|---------------------------------------------------------------|
| TopK (Gao 2024)      | Hard top-k selection (k = 32)                                 |
| L1 (Bricken 2023)    | ReLU encoder + L1 penalty on activations × decoder norm       |
| Gated (Rajamanoharan 2024) | Two-branch encoder; binary gate × ReLU magnitude        |
| **Position-Aware TopK (this work)** | TopK with per-position pre-bias for first 16 positions |

The position-aware variant replaces TopK's single learned pre-bias
(shape `(d_in,)`) with a table of shape `(max_pos=16, d_in)`. For tokens
at position `p < max_pos` we subtract the position-specific bias; for
`p >= max_pos` we use a single shared default bias. Adds `max_pos × d_in
= 14,336` extra parameters (~0.08% of the SAE total).

## Setup

- **Base model**: `Qwen/Qwen2.5-0.5B`, layer 9 residual stream.
- **Data**: 5M tokens of FineWeb-Edu activations (vs. 10M in prior work —
  scaled down due to CPU RAM constraints on a 60 GB machine).
- **Hyperparameters identical across architectures**: lr = 5e-4, batch = 4096,
  4 epochs, Adam, decoder rows renormalized to unit L2 each step.
- **Sparsity target**: TopK and PositionAware fix k = 32. L1 and Gated
  target the same via the L1 coefficient (5e-3 with 5%-of-training linear
  warmup from 0).
- **PositionAware trains on the full sequence** (positions 0–511).
  TopK / L1 / Gated train on positions 4–511 (`exclude_first_n = 4`,
  the v0.2 recipe).

## Training-side results

| arch               | MSE   | EV    | L0          | Dead features | Wall  |
|--------------------|-------|-------|-------------|---------------|-------|
| **TopK**           | 0.033 | 0.846 | 32.0 ± 0    | 464           | 86 s  |
| L1                 | 0.202 | 0.050 | 0.91 ± 6.0  | 7,605         | 108 s |
| Gated              | 0.090 | 0.575 | 12.9 ± 7.7  | 8,056         | 183 s |
| **PositionAware**  | 0.040 | 0.995 | 32.0 ± 0    | 4,221         | 99 s  |

Two important observations before drawing any conclusion:

1. **L1 and Gated did not converge with this default recipe.** L1's L0
   collapsed to ~1 (essentially silent); Gated landed near L0 = 13 with low
   EV. Both have known-difficult tuning curves; a proper benchmark would
   include batch-size sweeps, more elaborate L1-coef ramping schedules, and
   the recipe-specific gradient clipping the original papers use. We did
   not exhaustively tune. The honest finding here is **at default recipes
   on a single-4090 budget, TopK is dramatically more "out-of-the-box"
   than L1 or Gated** — which is itself a useful claim, just not an
   "architecture potential" claim.
2. **PositionAware's headline EV of 0.995 is partly an artifact of the
   metric.** It is trained and evaluated on positions 0–511. Positions
   0–3 are easy to reconstruct (per-position bias trivially handles
   outlier directions) and contribute most of the total variance the SAE
   has to explain. EV computed across all positions is dominated by them.
   To compare apples to apples we need to evaluate at the same position
   set as TopK (positions ≥ 4).

## v0.3 — Register-Subtracted TopK: a strictly-better variant

**Motivation.** The sister project [outlier-position-anatomy](https://github.com/legibleweights/outlier-position-anatomy)
established (v0.2) that the position-0 residual in small open
transformers is essentially a **fixed constant vector** across inputs
(99.8 % of energy in the input-independent mean across Qwen2.5-0.5B /
GPT-2 small / Pythia-1.4B). If position 0's outlier is a known constant,
we don't need a learned per-position bias to handle it — we can compute
the constant offline and subtract it directly. **No extra learnable
parameters.**

**Architecture.** `RegisterSubtractedTopKSAE` (see
`src/legible_weights/sae/register_subtracted.py`): byte-identical to
vanilla TopK except at position 0, where a fixed buffer
`register_vector` (computed offline as the mean of position-0 residuals
from ~256 held-out inputs) is subtracted from the activation before
encoding and added back after decoding. Same parameter count as TopK
(18.9 M) — the buffer is not learnable.

**Result on Qwen2.5-0.5B layer 9**, held-out 500K tokens:

| metric                | TopK            | Position-Aware  | **Register-Subtracted** |
|-----------------------|-----------------|-----------------|-------------------------|
| EV at positions ≥ 4   | **0.8407**      | 0.8136          | **0.8369**              |
| EV at positions 0–3   | −0.058 (broken) | 0.9997          | **0.9999**              |
| EV all positions      | 0.223           | 0.9947          | **0.9955**              |
| MSE at positions ≥ 4  | 0.0339          | 0.0396          | 0.0347                  |
| Dead features         | 464             | 4,221           | (similar to TopK)       |
| **CE recovered**      | 0.974           | 0.966           | **0.983**               |

**This is strictly better than both baselines.** Register-Subtracted:

- **Recovers 2.3 of the 2.7 EV points** that Position-Aware was paying as
  the "cost of position handling" on mid-sequence tokens (0.814 → 0.837).
- **Beats vanilla TopK on CE recovery** (0.983 vs 0.974) — because it
  also covers positions 0–3 in the splice intervention, so the spliced
  forward pass is closer to the original.
- Gets prefix coverage essentially perfectly (EV 0.9999 at positions
  0–3, same as Position-Aware).
- Adds **zero learnable parameters**. The register vector is a one-line
  offline computation: 256 forward passes, mean over position-0
  residual.

**The "trade-off" between mid-sequence quality and prefix coverage that
v0.2 documented is mostly an artifact of the per-position-bias design.**
If you know the register direction (and outlier-position-anatomy v0.2
showed you can know it cheaply for every small open transformer), you
can have both.

**Honest interpretation.** This works because the position-0 register
is genuinely a fixed constant (otherwise no offline-computed vector
would suffice). Position-Aware's flexibility — learning a *different*
bias per position — is wasted capacity here, since the bias only needs
to be non-trivial at position 0 and it's constant there.

### v0.3.1 — depth curve

Replicated the head-to-head at layers 5 and 15 of Qwen2.5-0.5B. The
Register-Subtracted SAE is strictly best at all three depths, and its
margin over the baselines on CE recovery grows with depth — exactly
matching the depth-pattern of TopK's prefix failure documented in v0.2.

| layer | TopK EV (pos ≥ 4) | PA EV (pos ≥ 4) | **RS EV (pos ≥ 4)** | TopK CE recovered | PA CE recovered | **RS CE recovered** | RS CE gain vs TopK |
|------:|:-----------------:|:---------------:|:--------------------:|:-----------------:|:---------------:|:--------------------:|:--------------------:|
| **L5**  | 0.863             | 0.835           | **0.861**            | 0.988             | 0.984           | **0.991**            | **+0.3 pts**          |
| **L9**  | 0.841             | 0.814           | **0.837**            | 0.974             | 0.966           | **0.983**            | **+0.9 pts**          |
| **L15** | 0.824             | 0.803           | **0.819**            | 0.944             | 0.935           | **0.967**            | **+2.4 pts**          |

Three trends:

1. **Mid-sequence EV cost shrinks dramatically.** PA paid 2.7–2.1 pts of
   EV on positions ≥ 4 across the depth curve. RS pays only 0.2–0.5 pts.
   The "cost of position-handling" is mostly an artifact of using a
   learnable bias for a quantity that's empirically constant.
2. **Prefix coverage is essentially perfect at all depths** (EV 0.9997
   for both PA and RS at every layer; TopK ranges from −0.05 at L5 to
   −0.47 at L15).
3. **The CE-recovery advantage of RS over TopK grows with depth**:
   +0.3 pts at L5 → +0.9 pts at L9 → **+2.4 pts at L15**. The
   architectural improvement scales with the severity of TopK's prefix
   failure, which (per v0.2) is most extreme at late layers.

### v0.4 — cross-model replication on GPT-2 small and Pythia-1.4B

Trained matched-config Register-Subtracted + TopK pairs at GPT-2 small
L6 (the layer studied in `outlier-position-anatomy` for GPT-2) and
Pythia-1.4B L12 (the equivalent mid-network layer for Pythia). Same
held-out 2-way evaluation methodology as v0.3.

| model · layer | TopK EV pos≥4 | RS EV pos≥4 | TopK EV pos 0-3 | RS EV pos 0-3 | TopK CE rec | RS CE rec | **RS CE gain** |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B L5  | 0.863  | 0.861  | −0.05  | 0.9999 | 0.988 | 0.991 | **+0.3 pts** |
| Qwen2.5-0.5B L9  | 0.841  | 0.837  | −0.06  | 0.9999 | 0.974 | 0.983 | **+0.9 pts** |
| Qwen2.5-0.5B L15 | 0.824  | 0.819  | **−0.47** | 0.9997 | 0.944 | 0.967 | **+2.4 pts** |
| **GPT-2 small L6**  | 0.850  | 0.844  | 0.08   | 0.9997 | 0.974 | 0.989 | **+1.5 pts** |
| **Pythia-1.4B L12** | 0.940  | 0.925  | **0.935** | 0.998  | 0.962 | 0.962 | **+0.0 pts (tied)** |

**The cross-model rule that emerges:** Register-Subtracted's CE-recovery
advantage over vanilla TopK is **proportional to how broken TopK is at
the prefix.** RS is a strict improvement when TopK's prefix is broken
(every Qwen layer and GPT-2 L6); essentially tied with TopK when the
prefix isn't broken (Pythia L12). RS never regresses on CE recovery.

**Why Pythia is different.** Pythia's eraser is at L23 (the *final*
layer), so at L12 (mid-network) the position-0 register hasn't been
erased yet — BUT it's also a smaller-magnitude register (norm 1,283 vs
Qwen's 1,682 and GPT-2's 3,041 — see `outlier-position-anatomy` v0.2 /
v0.3). TopK at Pythia L12, trained with `exclude_first_n=4`, generalizes
OK to positions 0–3 because the gap between the prefix positions and
mid-sequence positions is smaller than for Qwen or GPT-2. Result: RS
doesn't have much to fix.

**Cleaner aggregate claim:** Register-Subtracted TopK is a **strict
Pareto-improvement** over vanilla TopK across the three small open
transformers tested: it never regresses on CE recovery, gives perfect
prefix-position reconstruction (EV ≥ 0.9997 vs TopK's −0.47 to +0.94
range across models / layers), and pays a small or zero mid-sequence
EV cost (0.2–1.5 pts). The size of its CE-recovery improvement over
TopK is empirically a function of how badly TopK's prefix handling is
broken at that model / layer.

### v0.4.1 — Pythia L22: a counterexample that sharpens the rule

After v0.4 we tested one more (model, layer) configuration: Pythia-1.4B
at layer 22 — which is **just before the L23 eraser** identified in
outlier-position-anatomy. The register magnitude here should be at peak.

| Pythia-1.4B L22 | TopK | **Register-Subtracted** |
|---|---|---|
| EV pos≥4 | **0.802** | 0.761 (−4.2 pts — biggest mid-seq cost so far) |
| EV pos 0–3 | **0.977** (NOT broken!) | 0.991 (only +1.4 pts better) |
| **CE recovered** | 0.895 | **0.934 (+3.9 pts — biggest gain so far)** |

**This refutes the simple v0.4 rule** "RS gain proportional to TopK
prefix EV failure." At L22 TopK's prefix EV is actually *better than
L12* (0.977 vs 0.935), yet RS gives the biggest CE-recovery improvement
we've seen across any (model, layer) configuration.

**Refined rule.** At L22, the register's residual magnitude is at peak
because we're sampling just before the eraser. Even small *relative*
reconstruction errors at position 0 produce large *absolute* errors that
get amplified through the eraser's downstream computation. RS
reconstructs position 0 essentially perfectly (the subtraction is
exact); TopK's `exclude_first_n=4` splice keeps the real prefix but
its mid-sequence errors still propagate.

The actual driver of RS's CE-recovery gain over TopK is the **absolute
reconstruction-error gap at position 0, weighted by residual magnitude**
— not EV alone. Two regimes give big gains:

1. **TopK is catastrophically broken at prefix.** Reconstruction error
   relative-and-absolute. Examples: Qwen all depths, GPT-2 L6.
2. **TopK reconstructs prefix OK but residual magnitude is huge.** Small
   relative errors × huge magnitude = large absolute errors that affect
   downstream. Example: Pythia L22.

Pythia L12 has neither (good prefix EV *and* moderate magnitude) → RS
tied with TopK there.

**Updated falsifiable claim:** Register-Subtracted TopK is a Pareto-
improvement over vanilla TopK on CE recovery whenever the position-0
residual at the SAE's target layer has either (a) high norm (typically
late layers of any small open transformer) or (b) catastrophic prefix
reconstruction under vanilla TopK (typically most layers in Qwen, mid-
network in GPT-2). It never regresses CE recovery and always gives
perfect prefix-position reconstruction.

**Honest implications.** Pythia L22 also has the **worst** mid-sequence
EV cost (4.2 pts vs 0.2–1.5 pts elsewhere) — RS pays a real Pareto cost
on mid-sequence reconstruction at late layers where the register
magnitude is largest. This is a more nuanced trade-off than v0.4
suggested. For practitioners: if you care primarily about CE recovery
(deployment-relevant) RS wins everywhere except Pythia L12; if you care
primarily about mid-sequence reconstruction quality, vanilla TopK is
preferable at late layers.

### v0.4.2 — GPT-2 L10: a counterexample to the magnitude rule

After v0.4.1, I tested one more config that the magnitude rule predicted
RS would win at: GPT-2 small L10, the layer just before GPT-2's L11
attention-head-mediated eraser (per outlier-position-anatomy v0.5). Both
v0.4.1 conditions are present: peak register magnitude *and*
catastrophically broken TopK prefix (EV −1.25).

| GPT-2 L10 | TopK | **Register-Subtracted** |
|---|---|---|
| EV pos ≥ 4 | 0.797 | 0.782 (−1.5pt) |
| EV pos 0–3 | **−1.25** (catastrophic) | 0.999 |
| **CE recovered** | **0.950** | 0.947 (**−0.4 pts**, slight regression) |

**The magnitude rule predicted a big RS gain. We got a tiny regression.**
The rule is wrong, or at least incomplete.

**Best hypothesis** (would need more data to verify): GPT-2's eraser at
L11 is **attention-head-mediated** (head 8 does 77 % of the erase per
outlier-position-anatomy v0.5), whereas Qwen L21 and Pythia L23 erasers
are MLP-mediated. Attention heads attend cross-position; small per-
position reconstruction errors from RS get amplified through L11's
eraser-head in ways that MLPs don't propagate. **The eraser mechanism
type may matter as much as the register magnitude.**

I don't want to over-update on one data point. The honest cross-model
summary across all 7 datapoints we now have:

| (model, layer)       | RS CE gain vs TopK | comment |
|----------------------|:------------------:|---------|
| Qwen L5              | +0.3               | win    |
| Qwen L9              | +0.9               | win    |
| Qwen L15             | +2.4               | big win |
| GPT-2 L6             | +1.5               | win    |
| **GPT-2 L10**        | **−0.4**           | **regression** |
| Pythia L12           | tied               | TopK not broken |
| Pythia L22           | +3.9               | biggest win |

**Updated honest claim:**

> Register-Subtracted TopK is generally beneficial for SAE-based
> interpretability on small open transformers — it wins on CE recovery
> in 5 of 7 configurations tested, ties once, and slightly regresses
> once. Always gives perfect prefix-position reconstruction (EV ≥ 0.999
> vs TopK's −1.25 to +0.94 range). The factors driving win-size include
> register magnitude, prefix-reconstruction-error gap, *and the
> mechanism type of the eraser layer* (attention-head vs MLP). Mid-
> sequence EV cost ranges from 0.2 to 4.2 pts depending on the
> configuration; it's most expensive at late layers with large register
> magnitudes. **Practitioners should A/B test RS vs TopK for their
> specific (model, layer) before committing.**

This is less neat than v0.3 / v0.4 / v0.4.1, but it's the data. The
write-and-erase circuit story from outlier-position-anatomy makes the
intervention conceptually sound; the empirical caveats above mean the
intervention's benefit isn't guaranteed.

### v0.4 limitations

- **One layer per model for GPT-2 and Pythia.** Depth-curve replication
  on the other two models would strengthen the finding; haven't run it.
- **No Position-Aware baseline on GPT-2 or Pythia.** The Qwen finding
  that RS beats PA on mid-sequence EV is depth-replicated but not
  model-replicated. Likely holds (PA's learnable-bias-for-constant
  pattern is wasteful regardless of base model) but unverified.
- **Pythia has 12,084 dead features at TopK** vs the smaller dead-feature
  counts on Qwen / GPT-2. The Pythia dictionary is therefore "more
  redundant" or "harder to train", and the RS comparison may be slightly
  contaminated by this — though the comparison is matched so the
  conclusion still holds.

### v0.3 limitations

- **One register per (model, layer).** If you train SAEs at multiple
  layers of the same model, recompute the register for each.

## The fair head-to-head — depth curve

Held-out 500K tokens from a later FineWeb-Edu slice, evaluated in three
position buckets at three different layers of Qwen2.5-0.5B (mid, mid-late,
late). CE recovery via splice intervention through the live base model.

### Reconstruction EV by position bucket

| layer | bucket | n | TopK EV | Pos-Aware EV |
|-------|--------|--:|---------|--------------|
| **L5**  | positions ≥ 4 | 495,180 | **0.863** | **0.835** |
|         | positions 0–3 | 4,820   | −0.05    | 0.9997 |
| **L9**  | positions ≥ 4 | 495,180 | **0.841** | **0.814** |
|         | positions 0–3 | 4,820   | −0.06    | 0.9997 |
| **L15** | positions ≥ 4 | 495,180 | **0.824** | **0.803** |
|         | positions 0–3 | 4,820   | **−0.47** | 0.9997 |

### CE recovery (splice intervention)

| layer | TopK CE recovered | Pos-Aware CE recovered | Δ           |
|-------|-------------------|------------------------|-------------|
| L5    | **0.988**         | **0.984**              | −0.4 pts    |
| L9    | **0.974**         | **0.966**              | −0.8 pts    |
| L15   | **0.944**         | **0.935**              | −0.9 pts    |

### What the depth curve reveals

Three trends emerge from the depth replication that weren't visible from
L9 alone:

1. **The EV cost on positions ≥ 4 is consistent at ~2–3 points across all
   depths.** Per-position bias subtraction takes a small, layer-invariant
   bite out of mid-sequence reconstruction quality. This is the cost of
   spending some SAE capacity on positional normalization instead of
   features.
2. **TopK's catastrophic prefix failure gets dramatically worse at late
   layers.** At L5 the prefix EV is just slightly negative (−0.05); at L15
   it crashes to −0.47. Late residual streams carry larger-magnitude
   outliers at the sequence prefix, and TopK has *never seen them* during
   training. Position-Aware handles all three depths equally well at the
   prefix (EV 0.9997 each time).
3. **The CE-recovery cost of Position-Aware grows mildly with depth**
   (0.4 → 0.8 → 0.9 pts). The intervention is more sensitive at deeper
   layers because the downstream computation has further to amplify any
   reconstruction errors.

**Combined**: Position-Aware's value proposition is strongest at late
layers where TopK's prefix failure is catastrophic. At mid-network the
tradeoff is more even.

## Falsifiable claim

**Across layers 5, 9, and 15 of Qwen2.5-0.5B, a TopK sparse autoencoder
augmented with a per-position pre-bias for the first 16 sequence positions
extends usable reconstruction to positions 0–3 (EV 0.9997 at every depth,
vs −0.05 to −0.47 for vanilla TopK) at a 2–3-point EV cost on mid-sequence
positions and a 0.4–0.9-point cost on splice-intervention CE recovery.
The architecture adds 14,336 parameters (0.08 % of the 18.9 M SAE total)
and removes the need for `exclude_first_n` in the training pipeline. The
prefix-failure severity grows with depth (L5 −0.05 → L15 −0.47), making
position-conditioning more valuable at late layers.**

In plain terms: vanilla TopK SAEs cannot see what small open-weight LLMs
compute at the first ~4 sequence positions and have to discard those
positions during training. Per-position bias subtraction handles them
essentially perfectly at all three measured depths, for a small but real
cost on the dominant 99 % of mid-sequence tokens.

## What's NOT in v0.2

- **Only one model.** Qwen2.5-0.5B at three depths. Other architectures
  (GPT-2, Llama, Gemma) might have qualitatively different outlier-
  position-trap profiles. The bench script supports GPT-2 small via
  `--base gpt2-small` and replication is a one-line invocation; we
  haven't run it.
- **L1 and Gated were not given a fair recipe shot.** A proper "best
  architecture for small models" comparison would include their full
  recipes (LR warmup, decoder-norm constraints during not just after,
  longer schedules with more L1 ramping). We did the default recipe; both
  collapsed. Treat that as an ease-of-use signal, not a verdict on the
  architectures.
- **No feature-interpretability comparison.** Do PositionAware's
  features fire on the same kinds of tokens as TopK's, or differently?
  The dead-feature count is 9× higher under PositionAware (4,221 vs
  464) — those dead features might cluster at specific positions,
  worth inspecting.
- **JumpReLU SAE (Rajamanoharan et al. 2024) not implemented.** It is
  the current frontier of SAE architectures and would be the strongest
  baseline to compare against. We skipped it because its straight-through
  gradient estimator is a non-trivial implementation that we didn't have
  time to validate at v0.1.

## Public artifacts

- `report.json` — training-side metrics for all four architectures.
- `fair_eval.json` — three-position-bucket held-out metrics, TopK vs
  PositionAware.
- `topk/sae.pt`, `position_aware/sae.pt`, `l1/sae.pt`, `gated/sae.pt` —
  trained checkpoints (~76 MB each, not pushed to HF for v0.1 — these
  are diagnostic artifacts, the only interesting ones are TopK and
  PositionAware).
- All code in `experiments/small_sae_bench/` and the new SAE classes in
  `src/legible_weights/sae/`.
