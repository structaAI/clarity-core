# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any

"""
Adaptive PSWA Bridge

Here we can extract both coarse and fine details and this Pseudo-Shifted Window Attention
is a dynamic compute rather than a static one

--Parameters--
- @dim: int: Dimension of Input Features
- @num_scales: int: Number of 

--Returns--
- torch.Tensor: Refined latent features with adaptively injected high-frequency details.

This module upgrades the static bridge into an intelligent, restoration-aware engine:

1. Multi-Scale Frequency Decomposition: Unlike static filters, this module employs a 
   pyramid of branches (f_i) with varying bottleneck ratios (D -> D/2^i). This allows 
   the model to simultaneously capture fine, medium, and coarse textural nuances 
   across the latent manifold.

2. Degradation-Aware Gating: By processing the tripartite fused embedding (Time, 
   Type, and Severity), the module generates a dynamic Softmax gate (g). This gate 
   acts as a neural router, prioritizing specific frequency scales based on the 
   detected corruption (e.g., favoring coarse scales for heavy blur and fine scales 
   for stochastic noise).

3. Learned Fidelity Fusion: The high-frequency components are combined using 
   learnable fusion weights (λ) and conditioned gates. This ensures that the 
   additive residual is mathematically tailored to the specific restoration task, 
   preventing artifacts and ensuring "Authentic" reconstruction fidelity.
"""
class AdaptivePSWABridge(nn.Module):
  def __init__(self, dim: int, num_scales: int = 3)-> None:
    super().__init__()

    self.dim: int = dim
    self.num_scales: int = num_scales

    self.frequency_branches: nn.ModuleList = nn.ModuleList([
      nn.Sequential(
        nn.Linear(dim, dim // 2**i),
        nn.GELU(),
        nn.Linear(dim // 2**i, dim)
      )for i in range(num_scales)
    ])

    self.degradation_encoder: nn.Sequential = nn.Sequential(
      nn.Linear(dim, dim // 2),
      nn.GELU(),
      nn.Linear(dim // 2, num_scales),
      nn.Softmax(dim=-1)
    )

    self.fusion_weights: nn.Parameter = nn.Parameter(torch.ones(num_scales) / num_scales)
    self.norm: nn.LayerNorm = nn.LayerNorm(dim)
  
  def forward(self, x: torch.Tensor, t_embed: torch.Tensor)-> torch.Tensor:
    # Normalize and extract features
    x_norm = self.norm(x)
    multi_scale_features: list = [branch(x_norm) for branch in self.frequency_branches]

    # Compute Degradation Aware Gates
    gates: Any = self.degradation_encoder(t_embed) # Shape: [B, 3]: Batch Size, Number of Channels (default: 3)

    # Stack the Features: [B, 3, N, D]
    stacked_featues = torch.stack(multi_scale_features, dim=1) # Batch Size (B), No. of Channels (default: 3), 
    
    # Combine learned weights and dynamic gates into [B, 3, 1, 1]
    # This allows broadcasting across the N and D dimensions
    weighted_gates = (gates * self.fusion_weights).unsqueeze(-1).unsqueeze(-1)

    # Compute the weighted sum across scaled dimensions (dim-1)
    fused = (stacked_featues * weighted_gates).sum(dim=1)

    for i, (weight, gate, feature) in enumerate(zip(self.fusion_weights, gates.unbind(-1), multi_scale_features)):
      fused += weight * gate.unsqueeze(1).unsqueeze(2) * feature

    # Return with Residual Value Added to Original Tensor
    return x + fused