"""
AdaptivePSWABridge — Degradation-Aware Multi-Scale Feature Refinement.

This module upgrades the static PSWABridge into an intelligent,
restoration-aware engine with three design principles:

1. Multi-Scale Frequency Decomposition
   A pyramid of branches with varying bottleneck ratios (D → D/2^i) captures
   fine, medium, and coarse textural nuances across the latent manifold.

2. Degradation-Aware Gating
   The time/type/severity conditioning vector is encoded into a Softmax gate
   that acts as a neural router, prioritising frequency scales appropriate for
   the detected corruption (heavy blur → coarse; stochastic noise → fine).

3. Learned Fidelity Fusion
   Per-scale learnable weights (λ) are combined with dynamic gates to produce
   a single residual that is mathematically tailored to the restoration task.

Parameters
----------
dim : int
    Input/output feature dimensionality.
num_scales : int
    Number of frequency decomposition branches (default: 3).
"""

import torch
import torch.nn as nn
from typing import Any


class AdaptivePSWABridge(nn.Module):
  def __init__(self, dim: int, num_scales: int = 3) -> None:
    super().__init__()

    self.dim: int = dim
    self.num_scales: int = num_scales

    # Multi-scale frequency branches: D → D/2^i → D
    self.frequency_branches: nn.ModuleList = nn.ModuleList([
      nn.Sequential(
        nn.Linear(dim, max(dim // (2 ** i), 1)),
        nn.GELU(),
        nn.Linear(max(dim // (2 ** i), 1), dim),
      )
      for i in range(num_scales)
    ])

    # Degradation encoder: maps conditioning signal to per-scale Softmax gates
    self.degradation_encoder: nn.Sequential = nn.Sequential(
      nn.Linear(dim, dim // 2),
      nn.GELU(),
      nn.Linear(dim // 2, num_scales),
      nn.Softmax(dim=-1),
    )

    # Learnable per-scale fusion weights (initialised uniform)
    self.fusion_weights: nn.Parameter = nn.Parameter(
      torch.ones(num_scales) / num_scales
    )
    self.norm: nn.LayerNorm = nn.LayerNorm(dim)

  def forward(self, x: torch.Tensor, t_embed: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    x : torch.Tensor
        Shape [B, N, D] — token feature sequence.
    t_embed : torch.Tensor
        Shape [B, D] — conditioning vector (time / degradation embedding).

    Returns
    -------
    torch.Tensor
        Shape [B, N, D] — refined features with residual connection.
    """
    x_norm = self.norm(x)

    # Compute per-branch features: list of [B, N, D]
    multi_scale_features = [branch(x_norm) for branch in self.frequency_branches]

    # Degradation-aware gates: [B, num_scales]
    gates: torch.Tensor = self.degradation_encoder(t_embed)

    # Stack branches: [B, num_scales, N, D]
    stacked = torch.stack(multi_scale_features, dim=1)

    # Combine learned weights and dynamic gates → [B, num_scales, 1, 1]
    # Normalise fusion_weights with softmax so they sum to 1
    normalised_weights = torch.softmax(self.fusion_weights, dim=0)
    weighted_gates = (gates * normalised_weights).unsqueeze(-1).unsqueeze(-1)

    # Single weighted sum across the scale dimension → [B, N, D]
    fused = (stacked * weighted_gates).sum(dim=1)

    return x + fused