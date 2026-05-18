# Small-SAE Benchmark: TopK / L1 / Gated vs. Position-Aware TopK

**Date:** 2026-05-18 (v0.1)

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

## The fair head-to-head

Held-out 500K tokens from a later FineWeb-Edu slice, evaluated in three
position buckets:

| position bucket          | n        | TopK             | PositionAware    |
|--------------------------|---------:|------------------|------------------|
| **positions ≥ 4**        | 495,180  | EV **0.841**, MSE 0.034 | EV **0.814**, MSE 0.040 |
| positions 0–3            | 4,820    | EV **−0.07**, MSE **632** | EV **0.9997**, MSE 0.164 |
| all positions            | 500,000  | EV 0.21, MSE 6.13      | EV 0.995, MSE 0.041    |

This is the clean result:

- On positions ≥ 4 (where TopK was trained), the two architectures are
  **essentially equivalent**: TopK 0.841 vs PositionAware 0.814 — a 3-point
  EV gap, with TopK slightly ahead. Some of PositionAware's expressive
  capacity is being spent on per-position biases instead of features.
- On positions 0–3 (the outlier prefix), TopK is **catastrophic** (negative
  EV, MSE > 600 — it has literally never seen these inputs and the
  activations there have 50–100× larger norms than mid-sequence).
  PositionAware reconstructs them essentially perfectly.

## Falsifiable claim

**On Qwen2.5-0.5B layer 9, a TopK sparse autoencoder augmented with a
per-position pre-bias for the first 16 sequence positions extends usable
reconstruction to positions 0–3 (improving EV from −0.07 to 0.9997) at a
3-point EV cost on mid-sequence positions (0.841 → 0.814). The architecture
adds 14,336 parameters (0.08% of the 18.9 M total) and removes the need
for `exclude_first_n` in the training pipeline.**

In plain terms: vanilla TopK can't see what the model is doing at the
sequence prefix and has to discard those positions. Per-position bias
subtraction handles them for nearly zero cost. The tradeoff is real but
small.

## What's NOT in v0.1

- **Only one model, one layer.** Qwen2.5-0.5B at L9. The outlier-position
  trap might be more or less severe at other depths or in other
  architectures.
- **L1 and Gated were not given a fair recipe shot.** A proper "best
  architecture for small models" comparison would include their full
  recipes (LR warmup, decoder-norm constraints during not just after,
  longer schedules with more L1 ramping). We did the default recipe; both
  collapsed. Treat that as an ease-of-use signal, not a verdict on the
  architectures.
- **No CE recovery eval here.** The benchmark reports training-side and
  held-out reconstruction metrics. Adding CE recovery via splice
  intervention is straightforward (we have the `eval/recovery.py` code
  from the other projects) and a natural v0.2.
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
