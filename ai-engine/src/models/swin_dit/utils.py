import torch
from typing import Tuple

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
  B, H, W, C = x.shape
  x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
  windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
  return windows, H, W

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int)-> torch.Tensor:
  B = int(windows.shape[0] / (H * W / window_size / window_size))
  C = windows.shape[-1]

  x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
  x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
  
  return x