"""L1-sparsity sparse autoencoder.

The original Anthropic-style SAE (Bricken et al. 2023). Features are
ReLU(W_enc x + b), and sparsity is enforced by an L1 penalty on the
feature activations in the loss. Unlike TopK, sparsity is *soft* — set by
the relative weighting of the reconstruction term against the L1 penalty —
and active feature count varies per token.

Used here as the historical baseline that TopK was designed to improve on
(no shrinkage bias) and that JumpReLU was designed to improve on (no per-
feature threshold). Included in the small-SAE benchmark because many
existing public SAEs and pipelines still default to L1.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class L1SAEConfig:
    d_in: int
    d_hidden: int
    l1_coef: float = 1e-3
    normalize_decoder: bool = True


class L1SAE(nn.Module):
    def __init__(self, cfg: L1SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Linear(cfg.d_in, cfg.d_hidden, bias=True)
        self.decoder = nn.Linear(cfg.d_hidden, cfg.d_in, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_in))
        self._init_weights_tied()

    def _init_weights_tied(self) -> None:
        with torch.no_grad():
            w = torch.randn(self.cfg.d_in, self.cfg.d_hidden)
            w /= w.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.decoder.weight.copy_(w)
            self.encoder.weight.copy_(w.T)
            self.encoder.bias.zero_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x - self.pre_bias))

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self.decoder(acts) + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon, acts

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        recon, acts = self(x)
        mse = (recon - x).pow(2).mean()
        # L1 weighted by decoder column norms (Anthropic convention): penalizes
        # the *effective* contribution of each feature to the reconstruction.
        dec_norms = self.decoder.weight.norm(dim=0)  # (d_hidden,)
        l1 = (acts.abs() * dec_norms).sum(dim=-1).mean()
        total = mse + self.cfg.l1_coef * l1
        return total, {"mse": mse.detach(), "l1": l1.detach()}

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        if not self.cfg.normalize_decoder:
            return
        norms = self.decoder.weight.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.data.div_(norms)
