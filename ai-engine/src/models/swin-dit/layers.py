import torch
import torch.nn as nn
import math
from typing import Tuple

class TimeStepEmbedding(nn.Module):
  def __init__(self, hidden_size: int, frequency_embedding_size: int = 256)-> None:
    super().__init__()

    self.hidden_size = hidden_size
    self.frequency_embedding_size = frequency_embedding_size

    self.mlp: nn.Sequential = nn.Sequential(
      nn.Linear(self.frequency_embedding_size, self.hidden_size),
      nn.SiLU(),
      nn.Linear(self.hidden_size, self.hidden_size),
    )
  
  def forward(self, x: torch.Tensor)-> torch.Tensor:
    half_dim = self.frequency_embedding_size // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, device=x.device)/half_dim)
    args = x[:, None].float() * freqs[None, :]
    x_freq = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    return self.mlp(x_freq)

class SwinPatchEmbed(nn.Module):
  def __init__(self, patch_size: int = 2, no_of_in_channels: int = 4, embed_dim: int = 768)-> None:
    super().__init__()

    self.projection: nn.Conv2d = nn.Conv2d(no_of_in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    self.norm: nn.LayerNorm = nn.LayerNorm(embed_dim)
  
  def forward(self, x: torch.Tensor)-> Tuple[torch.Tensor, Tuple[int, int]]:
    x = self.projection(x)
    _, _, H, W = x.shape

    x = x.flatten(2).transpose(1, 2)

    return self.norm(x), (H, W)
