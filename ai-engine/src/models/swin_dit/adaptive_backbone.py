import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Any

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
    multi_scale_features: list = [branch(self.norm(x)) for branch in self.frequency_branches]

    gates: Any = self.degradation_encoder(t_embed)

    fused = torch.zeros_like(x)

    for i, (weight, gate, feature) in enumerate(zip(self.fusion_weights, gates.unbind(-1), multi_scale_features)):
      fused += weight * gate.unsqueeze(1).unsqueeze(2) * feature

    return x + fused