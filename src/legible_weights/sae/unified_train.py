"""Unified training loop for all SAE variants in the small-SAE benchmark.

Dispatches on architecture (TopK / L1 / Gated / PositionAware) so that one
script can train any of them on the same activation buffer with the same
optimizer, learning rate, batch size, and number of epochs. Differences
between architectures (loss function, position input, hard vs soft
sparsity) are encapsulated inside each class.

The benchmark uses **matched sparsity at evaluation time**, not at training
time — i.e., we train each SAE with its natural recipe, then compare them
at whatever sparsity they settle at. For L1 / Gated this means tuning
l1_coef to land near L0 = 32 (the TopK k), with a default coefficient that
gets us close.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm


@dataclass
class UnifiedTrainConfig:
    batch_size: int = 4096
    lr: float = 5e-4
    n_epochs: int = 4
    log_every: int = 25
    device: str = "cuda"
    # L1-coef linear warmup for L1 / Gated SAEs. Over the first
    # `l1_warmup_frac * total_steps`, l1_coef ramps from 0 to its target.
    # Avoids features dying before they get a chance to learn.
    l1_warmup_frac: float = 0.05


def _needs_positions(sae) -> bool:
    return hasattr(sae, "encode") and "positions" in sae.encode.__code__.co_varnames


def train_unified(
    sae: torch.nn.Module,
    activations: torch.Tensor,        # (N, d_in) fp16 on CPU
    cfg: UnifiedTrainConfig,
    positions: torch.Tensor | None = None,   # (N,) int32 on CPU; required for PositionAware
) -> list[dict]:
    sae.to(cfg.device).train()
    opt = torch.optim.Adam(sae.parameters(), lr=cfg.lr)
    n = activations.shape[0]
    history: list[dict] = []

    needs_pos = _needs_positions(sae)
    if needs_pos and positions is None:
        raise ValueError("This SAE expects positions but none were provided")

    # Initialize pre-bias to the dataset mean — sample 100k rows rather than
    # casting the entire buffer to fp32 (which would allocate ~35 GB on 10M-row
    # buffers and trigger OOM).
    sample_n = min(100_000, n)
    sample = activations[:sample_n].float()
    mean_act = sample.mean(dim=0).to(cfg.device)
    total_var_sample = sample.var(dim=0).sum().item()
    del sample
    with torch.no_grad():
        if hasattr(sae, "pre_bias") and not needs_pos:
            sae.pre_bias.copy_(mean_act)
        elif hasattr(sae, "pre_bias_default"):
            sae.pre_bias_default.copy_(mean_act)
            sae.pre_bias_per_pos.data[:] = mean_act.unsqueeze(0)

    total_var = total_var_sample

    # Set up L1 warmup if applicable
    target_l1 = None
    if hasattr(sae, "cfg") and hasattr(sae.cfg, "l1_coef"):
        target_l1 = float(sae.cfg.l1_coef)
        sae.cfg.l1_coef = 0.0  # start ramp from 0
    total_steps = max(1, cfg.n_epochs * (n // cfg.batch_size))
    warmup_steps = max(1, int(cfg.l1_warmup_frac * total_steps))

    step = 0
    for epoch in range(cfg.n_epochs):
        perm = torch.randperm(n)
        pbar = tqdm(
            range(0, n - cfg.batch_size + 1, cfg.batch_size),
            desc=f"epoch {epoch + 1}/{cfg.n_epochs}",
        )
        for start in pbar:
            # L1 warmup: linearly ramp from 0 to target_l1 over warmup_steps
            if target_l1 is not None:
                sae.cfg.l1_coef = target_l1 * min(1.0, step / warmup_steps)

            idx = perm[start : start + cfg.batch_size]
            x = activations[idx].to(cfg.device, dtype=torch.float32)

            if needs_pos:
                pos = positions[idx].to(cfg.device).long()
                if hasattr(sae, "loss"):
                    loss, parts = sae.loss(x, pos)
                else:
                    recon, _ = sae(x, pos)
                    loss = (recon - x).pow(2).mean()
                    parts = {"mse": loss.detach()}
            else:
                if hasattr(sae, "loss"):
                    loss, parts = sae.loss(x)
                else:
                    recon, _ = sae(x)
                    loss = (recon - x).pow(2).mean()
                    parts = {"mse": loss.detach()}

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if hasattr(sae, "normalize_decoder_"):
                sae.normalize_decoder_()

            if step % cfg.log_every == 0:
                with torch.no_grad():
                    if needs_pos:
                        recon, acts = sae(x, pos)
                    else:
                        recon, acts = sae(x)
                    residual_var = (x - recon).var(dim=0).sum().item()
                    ev = 1.0 - residual_var / total_var
                    l0 = (acts > 0).float().sum(dim=-1).mean().item()
                history.append({
                    "step": step,
                    "epoch": epoch,
                    "loss": float(loss.item()),
                    "mse": float(parts.get("mse", loss).item()),
                    "explained_var": ev,
                    "l0": l0,
                })
                pbar.set_postfix(
                    mse=f"{parts.get('mse', loss).item():.4f}",
                    ev=f"{ev:.3f}",
                    l0=f"{l0:.1f}",
                )
            step += 1

    return history
