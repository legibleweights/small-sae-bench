# Small-SAE Benchmark: TopK / L1 / Gated vs. Position-Aware TopK

**Date:** 2026-05-18 (v0.2 — depth replication + CE recovery)

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
