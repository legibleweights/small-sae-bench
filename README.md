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

## Headline finding

On Qwen2.5-0.5B layer 9, held-out 500K tokens, **TopK vs Position-Aware TopK**:

| position bucket          | n        | TopK             | Position-Aware   |
|--------------------------|---------:|------------------|------------------|
| **positions ≥ 4**        | 495,180  | EV **0.841**     | EV **0.814**     |
| **positions 0–3**        | 4,820    | EV **−0.07** (MSE 632) | EV **0.9997** (MSE 0.16) |

Position-Aware TopK adds a per-position pre-bias for the first 16 positions
(14,336 extra params, 0.08 % of the SAE total) and **extends usable
reconstruction to the full sequence at a 3-point EV cost on mid-sequence
positions**. The standard `exclude_first_n` workaround is no longer needed.

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

**On Qwen2.5-0.5B layer 9, adding a per-position pre-bias for the first 16
sequence positions to a TopK SAE extends usable reconstruction to
positions 0–3 (EV −0.07 → 0.9997) at a 3-point EV cost on mid-sequence
positions (0.841 → 0.814).** The architecture adds 0.08 % to the parameter
count and eliminates the need for `exclude_first_n` in the training
pipeline.

## Part of

[legible-weights](https://github.com/legibleweights/legible-weights) — an
umbrella research thread on interpretability of small open-weight LLMs.

## License

MIT.
