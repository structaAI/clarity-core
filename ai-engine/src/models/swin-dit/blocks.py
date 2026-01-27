import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

class PSWABridge(nn.Module):
  def __init__(self, dim: int)-> None:
    super().__init__()
    self.high_frequency_extractor: nn.Sequential = nn.Sequential(
      nn.Linear(dim, dim //4),
      nn.GELU(),
      nn.Linear(dim // 4, dim)
    )

    self.scale: nn.Parameter = nn.Parameter(torch.ones(1) * 0.1)
    self.norm: nn.LayerNorm = nn.LayerNorm(dim)
  
  def forward(self, x: torch.Tensor)-> torch.Tensor:
    return x + self.high_frequency_extractor(self.norm(x)) * self.scale

class SwinBlock(nn.Module):
  def __init__(self, dim: int, num_heads: int, use_pswa_bridge: bool = False)-> None:
    super().__init__

    self.attention: nn.MultiheadAttention = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    self.norm1: nn.LayerNorm = nn.LayerNorm(dim)
    self.bridge: Optional[PSWABridge] = PSWABridge(dim) if use_pswa_bridge else None
  
  def forward(self, x: torch.Tensor, t_embed: torch.Tensor)-> torch.Tensor:
    res = x + t_embed.unsqueeze(1)

    x_norm = self.norm1(res)
    attention_output, _ = self.attention(x_norm, x_norm, x_norm)

    x = res + attention_output
    if self.bridge is not None:
      x = self.bridge(x)
    
    return x