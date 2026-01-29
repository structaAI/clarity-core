import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .utils import window_partition, window_reverse

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
  def __init__(self, dim: int, num_heads: int, window_size: int = 8, shifted: bool = False, use_pswa_bridge: bool = False)-> None:
    super().__init__()

    self.dim: int = dim
    self.num_heads: int = num_heads
    self.window_size: int = window_size
    self.shifted = shifted

    self.attention: nn.MultiheadAttention = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    self.norm1: nn.LayerNorm = nn.LayerNorm(dim)
    self.bridge: Optional[PSWABridge] = PSWABridge(dim) if use_pswa_bridge else None
  
  def forward(self, x: torch.Tensor, t_embed: torch.Tensor, H: int, W: int)-> torch.Tensor:
    res = x + t_embed.unsqueeze(1)
    x_norm = self.norm1(res)

    B, L, C = x_norm.shape
    x_grid = x_norm.view(B, H, W, C)

    if self.shifted:
      shift_size = self.window_size // 2
      x_grid = torch.roll(x_grid, shifts=(-shift_size, -shift_size), dims=(1, 2))

    x_windows, _, _= window_partition(x_grid, self.window_size) # Tensor, p_H, p_W
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

    attention_output, _ = self.attention(x_windows, x_windows, x_windows)

    attention_output = attention_output.view(-1, self.window_size, self.window_size, C)
    x = window_reverse(attention_output, self.window_size, H, W)

    if self.shifted:
      shift_size = self.window_size // 2
      x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

    x = x.view(-1, L, C)
    x = res + x

    if self.bridge is not None:
      x = self.bridge(x)

    return x