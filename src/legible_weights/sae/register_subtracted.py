"""Register-Subtracted TopK sparse autoencoder.

Bridges outlier-position-anatomy and small-sae-bench:

- outlier-position-anatomy v0.2 showed that the position-0 residual on
  small open transformers is essentially a fixed vector (cosine ≥ 0.9996
  across hundreds of inputs; 99.8 %+ of energy in the input-independent
  mean).
- small-sae-bench v0.2 showed that handling position 0 via a per-position
  learned bias (Position-Aware TopK) costs ~3 EV points on mid-sequence
  positions, presumably because some of the SAE's capacity is spent
  re-learning the per-position normalization.

If position 0's "outlier" is just a fixed constant, we don't need a
learned per-position bias for it. We can compute the constant offline
and *subtract it directly* before encoding at position 0, then use
vanilla TopK everywhere. No extra learnable parameters at all.

Architecture
------------

Identical to TopK, plus one extra (non-learnable) buffer
`register_vector` of shape (d_in,). At inference / training:

    x[positions == 0] -= register_vector
    z = encoder(x - pre_bias)
    ...

For positions other than 0 the architecture is byte-identical to TopK.
The hypothesis is that this gets PositionAware's prefix-coverage benefit
without paying its mid-sequence EV cost.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RegisterSubtractedSAEConfig:
    d_in: int
    d_hidden: int
    k: int
    normalize_decoder: bool = True


class RegisterSubtractedTopKSAE(nn.Module):
    def __init__(self, cfg: RegisterSubtractedSAEConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Linear(cfg.d_in, cfg.d_hidden, bias=True)
        self.decoder = nn.Linear(cfg.d_hidden, cfg.d_in, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_in))
        # NOT a Parameter — fixed buffer set from outside before training
        self.register_buffer("register_vector", torch.zeros(cfg.d_in))

        self._init_weights_tied()

    def _init_weights_tied(self) -> None:
        with torch.no_grad():
            w = torch.randn(self.cfg.d_in, self.cfg.d_hidden)
            w /= w.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.decoder.weight.copy_(w)
            self.encoder.weight.copy_(w.T)
            self.encoder.bias.zero_()

    def set_register(self, register: torch.Tensor) -> None:
        """Install the known position-0 register vector (offline-computed)."""
        assert register.shape == self.register_vector.shape, \
            f"register shape mismatch: {register.shape} vs {self.register_vector.shape}"
        self.register_vector.copy_(register)

    def _subtract_register(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        is_pos0 = (positions == 0)
        if is_pos0.any():
            x = x.clone()
            x[is_pos0] = x[is_pos0] - self.register_vector
        return x

    def encode(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        x = self._subtract_register(x, positions)
        pre = self.encoder(x - self.pre_bias)
        vals, idx = pre.topk(self.cfg.k, dim=-1)
        vals = torch.relu(vals)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, idx, vals)
        return acts

    def decode(self, acts: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        out = self.decoder(acts) + self.pre_bias
        is_pos0 = (positions == 0)
        if is_pos0.any():
            out = out.clone()
            out[is_pos0] = out[is_pos0] + self.register_vector
        return out

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x, positions)
        recon = self.decode(acts, positions)
        return recon, acts

    def loss(
        self, x: torch.Tensor, positions: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        recon, _ = self(x, positions)
        mse = (recon - x).pow(2).mean()
        return mse, {"mse": mse.detach()}

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        if not self.cfg.normalize_decoder:
            return
        norms = self.decoder.weight.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.data.div_(norms)
