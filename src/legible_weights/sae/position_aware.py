"""Position-aware TopK sparse autoencoder.

Motivation. In our v0.1 → v0.2 work on Qwen2.5-0.5B layer 9, naive TopK SAEs
allocated a large fraction of dictionary capacity to fitting the first few
sequence positions (positions 0–3) because the residual stream has anomalously
high norm there (attention-sink / outlier-position phenomenon). The v0.2 fix
was crude: skip those positions during training entirely. That leaves a hole:
the SAE has nothing to say about what the model is computing at the prefix.

Hypothesis. A SAE that *conditions* on position — specifically, that subtracts
a learned per-position bias from the activation before encoding — should be
able to normalize the prefix activations into the same range as mid-sequence,
so TopK selection happens on a comparable distribution everywhere. If true,
the SAE recovers prefix-position features instead of having to ignore them.

Implementation. Replace the single learned pre_bias (shape (d_in,)) with a
table pre_bias[pos] of shape (max_pos, d_in). For tokens at position p < max_pos
we subtract pre_bias[p]; for tokens at position p >= max_pos we use a single
shared default bias. This keeps the parameter count modest (max_pos * d_in for
small max_pos like 16) while giving the model explicit per-position normalization
exactly where the outlier-position trap lives.

Forward signature is the same as TopK SAE except it also takes `positions`
(int tensor of shape matching the batch leading dim). Loss is the same MSE
(no L1 needed — TopK enforces sparsity).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PositionAwareSAEConfig:
    d_in: int
    d_hidden: int
    k: int
    max_pos: int = 16  # how many distinct positions get their own bias
    normalize_decoder: bool = True


class PositionAwareTopKSAE(nn.Module):
    def __init__(self, cfg: PositionAwareSAEConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Linear(cfg.d_in, cfg.d_hidden, bias=True)
        self.decoder = nn.Linear(cfg.d_hidden, cfg.d_in, bias=False)
        # Per-position biases for the first max_pos positions plus one
        # shared "default" bias used for positions >= max_pos.
        self.pre_bias_per_pos = nn.Parameter(torch.zeros(cfg.max_pos, cfg.d_in))
        self.pre_bias_default = nn.Parameter(torch.zeros(cfg.d_in))

        self._init_weights_tied()

    def _init_weights_tied(self) -> None:
        with torch.no_grad():
            w = torch.randn(self.cfg.d_in, self.cfg.d_hidden)
            w /= w.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.decoder.weight.copy_(w)
            self.encoder.weight.copy_(w.T)
            self.encoder.bias.zero_()

    def _bias_for(self, positions: torch.Tensor) -> torch.Tensor:
        """positions: int tensor shape (N,) → biases shape (N, d_in)."""
        in_table = positions < self.cfg.max_pos
        clamped = positions.clamp(max=self.cfg.max_pos - 1).long()
        per_pos = self.pre_bias_per_pos[clamped]               # (N, d_in)
        default = self.pre_bias_default.unsqueeze(0).expand_as(per_pos)
        return torch.where(in_table.unsqueeze(-1), per_pos, default)

    def encode(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bias = self._bias_for(positions)
        pre = self.encoder(x - bias)
        vals, idx = pre.topk(self.cfg.k, dim=-1)
        vals = torch.relu(vals)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, idx, vals)
        return acts

    def decode(self, acts: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bias = self._bias_for(positions)
        return self.decoder(acts) + bias

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
