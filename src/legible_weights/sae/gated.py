"""Gated sparse autoencoder (Rajamanoharan et al. 2024).

Decouples the "which features are active" decision from "how active they
should be." Two parallel encoder branches share the underlying weight
matrix:

  pi_gate(x)  = W_gate^T x + b_gate     # gating decision (binary)
  pi_mag(x)   = W_mag^T x + b_mag       # magnitude

The forward output uses:

  f(x) = 1[pi_gate(x) > 0] * ReLU(pi_mag(x))

with W_mag = exp(r_mag) * W_gate (W_gate shared, r_mag a learned per-feature
log-scale). The loss adds an L1 penalty on the *gating-branch* pre-activations,
which removes the L1-shrinkage bias of vanilla L1 SAEs because the magnitude
branch is uncoupled from the sparsity penalty.

A small auxiliary reconstruction loss using only the gating branch
(treated as a soft gate ReLU(pi_gate)) trains b_gate, ensuring it
informatively selects features rather than collapsing.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GatedSAEConfig:
    d_in: int
    d_hidden: int
    l1_coef: float = 1e-3
    aux_coef: float = 1.0
    normalize_decoder: bool = True


class GatedSAE(nn.Module):
    def __init__(self, cfg: GatedSAEConfig):
        super().__init__()
        self.cfg = cfg
        self.W_gate = nn.Parameter(torch.empty(cfg.d_in, cfg.d_hidden))
        self.r_mag = nn.Parameter(torch.zeros(cfg.d_hidden))
        self.b_gate = nn.Parameter(torch.zeros(cfg.d_hidden))
        self.b_mag = nn.Parameter(torch.zeros(cfg.d_hidden))
        self.decoder = nn.Linear(cfg.d_hidden, cfg.d_in, bias=False)
        self.pre_bias = nn.Parameter(torch.zeros(cfg.d_in))
        self._init_weights_tied()

    def _init_weights_tied(self) -> None:
        with torch.no_grad():
            w = torch.randn(self.cfg.d_in, self.cfg.d_hidden)
            w /= w.norm(dim=0, keepdim=True).clamp_min(1e-8)
            self.decoder.weight.copy_(w)
            self.W_gate.copy_(w)
            self.b_gate.zero_()
            self.b_mag.zero_()
            self.r_mag.zero_()

    def _pre_acts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = x - self.pre_bias
        pi_gate = z @ self.W_gate + self.b_gate
        W_mag = torch.exp(self.r_mag) * self.W_gate
        pi_mag = z @ W_mag + self.b_mag
        return pi_gate, pi_mag

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pi_gate, pi_mag = self._pre_acts(x)
        gate = (pi_gate > 0).to(pi_mag.dtype)
        return gate * torch.relu(pi_mag)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self.decoder(acts) + self.pre_bias

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon, acts

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        pi_gate, pi_mag = self._pre_acts(x)
        gate = (pi_gate > 0).to(pi_mag.dtype)
        acts = gate * torch.relu(pi_mag)
        recon = self.decode(acts)
        mse = (recon - x).pow(2).mean()

        # L1 on the *gating* pre-activations (softened by ReLU so it differentiates)
        # — penalizes features for being chosen, not for being big.
        dec_norms = self.decoder.weight.norm(dim=0)
        l1 = (torch.relu(pi_gate) * dec_norms).sum(dim=-1).mean()

        # Auxiliary reconstruction using the gating branch alone, which forces
        # b_gate to carry meaningful information instead of collapsing to a
        # threshold that's just a function of pi_mag.
        aux_acts = torch.relu(pi_gate)
        # Use a detached copy of the decoder (no decoder grads from aux task)
        aux_recon = aux_acts @ self.decoder.weight.detach().T + self.pre_bias.detach()
        aux = (aux_recon - x).pow(2).mean()

        total = mse + self.cfg.l1_coef * l1 + self.cfg.aux_coef * aux
        return total, {"mse": mse.detach(), "l1": l1.detach(), "aux": aux.detach()}

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        if not self.cfg.normalize_decoder:
            return
        norms = self.decoder.weight.data.norm(dim=0, keepdim=True).clamp_min(1e-8)
        self.decoder.weight.data.div_(norms)
