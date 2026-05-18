"""TopK sparse autoencoder for residual-stream activations.

Architecture follows Gao et al. 2024 ("Scaling and evaluating sparse
autoencoders"): a single hidden layer with a hard TopK activation. TopK is
preferred over L1 because it has no shrinkage bias and a cleaner sparsity
guarantee (exactly k active features per token).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SAEConfig:
    d_in: int
    d_hidden: int
    k: int
    normalize_decoder: bool = True


class TopKSAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
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

    def encode_pre(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x - self.pre_bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = self.encode_pre(x)
        vals, idx = pre.topk(self.cfg.k, dim=-1)
        vals = torch.relu(vals)
        acts = torch.zeros_like(pre)
        acts.scatter_(-1, idx, vals)
        return acts

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self.decoder(acts) + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon, acts

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Renormalize decoder rows to unit L2 norm. Call after each optimizer step."""
        if not self.cfg.normalize_decoder:
            return
        norms = self.decoder.weight.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.data.div_(norms)
