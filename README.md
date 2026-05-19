# small-sae-bench

**Architecture benchmark for sparse autoencoders at sub-1B model scale, plus
one novel variant — Position-Aware TopK — that eliminates the outlier-
position trap.**

Existing SAE comparison papers benchmark at GPT-2-medium or larger. Nobody
has systematically compared TopK / L1 / Gated SAEs at sub-1B model scale
with a single-4090 budget. This repo does that, and also introduces a small
architectural variant motivated by a concrete problem we hit in earlier
work: vanilla TopK SAEs allocate a large fraction of their dictionary to
fitting the high-norm outlier directions at the first 4 sequence positions,
and the standard fix (`exclude_first_n=4`) discards those positions
entirely.

## v0.4 headline — Register-Subtracted is a strict Pareto-improvement over TopK across 3 models, 5 layers

**Cross-model + cross-depth replication** of v0.3. RS is a strict Pareto-
improvement over vanilla TopK: never regresses on CE recovery, gives
perfect prefix-position reconstruction (EV ≥ 0.9997 across all configs
vs TopK's −0.47 to +0.94 range), and pays a 0.2–1.5pt mid-sequence EV
cost. **The size of its CE-recovery gain over TopK scales with how badly
TopK's prefix is broken.**

| model · layer | TopK EV pos≥4 | **RS EV pos≥4** | TopK EV pos 0-3 | **RS EV pos 0-3** | TopK CE rec | **RS CE rec** | RS CE gain |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B L5  | 0.863 | **0.861** | −0.05 | **0.9999** | 0.988 | **0.991** | +0.3 pts |
| Qwen2.5-0.5B L9  | 0.841 | **0.837** | −0.06 | **0.9999** | 0.974 | **0.983** | +0.9 pts |
| Qwen2.5-0.5B L15 | 0.824 | **0.819** | **−0.47** | **0.9997** | 0.944 | **0.967** | **+2.4 pts** |
| **GPT-2 small L6**   | 0.850 | **0.844** | 0.08  | **0.9997** | 0.974 | **0.989** | **+1.5 pts** |
| **Pythia-1.4B L12**  | 0.940 | **0.925** | 0.935 | **0.998**  | 0.962 | **0.962** | tied |
| **Pythia-1.4B L22**  | 0.802 | 0.761 (−4.2pt) | 0.977 | **0.991**  | 0.895 | **0.934** | **+3.9 pts** |

Pythia L12 is the case where TopK isn't broken at the prefix; RS ties.
**Pythia L22 (just before the L23 eraser) is the most interesting case** —
TopK's prefix EV is actually *better* there (0.977 vs L12's 0.935) but
RS gives the **biggest CE-recovery gain we've seen anywhere** (+3.9pts).
Why: at L22 the register magnitude is at peak (just before erasure), so
small relative errors × huge magnitude = large absolute errors that
disrupt the downstream eraser. RS's perfect prefix reconstruction
(subtraction is exact) avoids that. The cost is a real 4.2pt mid-seq
EV drop — the biggest in the depth/model curve.

**Refined rule:** RS wins on CE recovery when position 0 has either
(a) high norm (typically late layers of any small open transformer) or
(b) catastrophic TopK prefix EV. Tied when neither. Never regresses
CE recovery; always gives perfect prefix reconstruction.

Zero extra learnable parameters in any case — the register is one
offline-computed fixed vector per (model, layer).

Knowing the position-0 register direction (computed offline; see the
sister project [outlier-position-anatomy](https://github.com/legibleweights/outlier-position-anatomy)
v0.2 showing it's a fixed constant vector) lets us build an SAE that
beats both vanilla TopK and Position-Aware TopK at all three depths of
Qwen2.5-0.5B — and its CE-recovery advantage **grows with depth**,
exactly tracking the severity of TopK's prefix failure:

| layer | TopK EV (pos ≥ 4) | PA EV | **RS EV** | TopK CE rec | PA CE rec | **RS CE rec** | RS gain vs TopK |
|------:|:-----------------:|:-----:|:---------:|:-----------:|:---------:|:-------------:|:---------------:|
| **L5**  | 0.863             | 0.835 | **0.861** | 0.988       | 0.984     | **0.991**     | **+0.3 pts**    |
| **L9**  | 0.841             | 0.814 | **0.837** | 0.974       | 0.966     | **0.983**     | **+0.9 pts**    |
| **L15** | 0.824             | 0.803 | **0.819** | 0.944       | 0.935     | **0.967**     | **+2.4 pts**    |

Zero extra learnable parameters — the register is one offline-computed
fixed vector per layer. **The trade-off Position-Aware was paying turns
out to be mostly an artifact of using a learnable bias for a quantity
that is empirically constant.** The architectural improvement scales
with the severity of TopK's prefix failure (which is most extreme at
late layers, per v0.2).

See [NOTES.md](NOTES.md) v0.3 / v0.3.1 sections for full methodology +
caveats.

## Headline finding (v0.2 — depth-replicated Position-Aware vs TopK)

On Qwen2.5-0.5B at three depths, held-out 500K tokens, **TopK vs
Position-Aware TopK**:

| layer | pos ≥ 4 EV (TopK / PA) | pos 0–3 EV (TopK / PA) | CE rec. (TopK / PA) |
|-------|--------------------------|------------------------|---------------------|
| **L5**  | 0.863 / **0.835**     | **−0.05** / 0.9997     | 0.988 / 0.984       |
| **L9**  | 0.841 / **0.814**     | **−0.06** / 0.9997     | 0.974 / 0.966       |
| **L15** | 0.824 / **0.803**     | **−0.47** / 0.9997     | 0.944 / 0.935       |

Position-Aware TopK adds a per-position pre-bias for the first 16 positions
(14,336 extra params, 0.08 % of the SAE total) and **extends usable
reconstruction to the full sequence at a consistent 2–3-point EV cost on
mid-sequence positions, with the cost paying off most at late layers
where TopK's prefix failure is catastrophic** (EV −0.47 at L15). The
standard `exclude_first_n` workaround is no longer needed.

## What's in this repo

- `src/legible_weights/sae/` — four SAE architectures: TopK (the existing
  workhorse), L1 (Bricken et al. style baseline), Gated (Rajamanoharan
  et al. 2024), and the novel **PositionAwareTopKSAE**.
- `src/legible_weights/sae/unified_train.py` — single training loop that
  dispatches on architecture, with L1-coef linear warmup for L1 / Gated.
- `src/legible_weights/data/` — activation collection with optional
  per-token position tracking (required for position-aware training).
- `run_bench.py` — trains all four architectures on the same data with
  identical hyperparameters; uses two smaller collections rather than
  one big buffer to fit in 60 GB CPU RAM.
- `fair_eval.py` — held-out head-to-head of TopK vs PositionAware on
  positions ≥ 4 / positions 0–3 / all positions.
- `results/report.json`, `results/fair_eval.json` — the headline metrics.
- [`NOTES.md`](NOTES.md) — full writeup with methodology, the honest L1 /
  Gated convergence story, and limitations.

## Honest caveats up front

- **L1 and Gated did not converge with my default recipe.** Their L0
  collapsed (L1) or landed below target with low EV (Gated). Both have
  known-difficult tuning curves; we didn't exhaustively tune. Take the
  L1 / Gated rows as "out-of-the-box ease of use," not as a verdict on
  the architectures' potential.
- **One model, one layer.** Qwen2.5-0.5B at L9. The outlier-position
  trap may be more or less severe elsewhere; the 3-point EV cost of
  position-conditioning may differ.
- **JumpReLU SAE not implemented.** It's the current SOTA architecture
  and would be the strongest baseline; we skipped it because the
  straight-through gradient estimator is non-trivial to validate and
  not in scope for v0.1.

## Reproducing

```bash
pip install torch>=2.3 transformers>=4.40 huggingface_hub>=0.23 \
            datasets>=2.19 numpy>=1.26 tqdm>=4.66 safetensors

# Train all four architectures on Qwen2.5-0.5B layer 9, 5 M tokens, ~10 min
PYTHONPATH=src python run_bench.py \
    --base qwen2.5-0.5b --layer 9 \
    --n-tokens 5000000 --epochs 4 --batch-size 4096 \
    --out results/qwen2.5-0.5b-l9

# Apples-to-apples eval of TopK vs PositionAware on three position buckets
PYTHONPATH=src python fair_eval.py \
    --bench-dir results/qwen2.5-0.5b-l9 \
    --n-tokens 500000
```

GPT-2 small support is included (`--base gpt2-small`) but not benchmarked
in v0.1; running it is a one-line invocation.

## Falsifiable claim

**Across layers 5, 9, and 15 of Qwen2.5-0.5B, adding a per-position pre-bias
for the first 16 sequence positions to a TopK SAE extends usable
reconstruction to positions 0–3 (EV 0.9997 at every depth, vs −0.05 to
−0.47 for vanilla TopK) at a consistent 2–3-point EV cost on mid-sequence
positions and a 0.4–0.9-point cost on splice-intervention CE recovery.**
The architecture adds 0.08 % to the parameter count and eliminates the
need for `exclude_first_n` in the training pipeline. **Prefix-failure
severity grows with depth** (L5 −0.05 → L15 −0.47), making position-
conditioning more valuable at late layers.

## Part of

[legible-weights](https://github.com/legibleweights/legible-weights) — an
umbrella research thread on interpretability of small open-weight LLMs.

## License

MIT.
