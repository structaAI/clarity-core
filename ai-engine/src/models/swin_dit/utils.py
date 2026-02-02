# Imports
import torch
from typing import Tuple

""" 
Utility Functions for SwinDiT Model 

1. Window Partition: (Partitions Input Tensor into Smaller Windows)
--Parameters--
- @x: torch.Tensor: Input Tensor of shape (B, H, W, C)
- @window_size: int: Size of the Window to Partition the Tensor
--Returns--
- torch.Tensor: Partitioned Windows of shape (num_windows*B, window_size, window_size, C)
- int: Original Height (H)
- int: Original Width (W)

"""

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, int, int]:
  B, H, W, C = x.shape 
  x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
  windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
  return windows, H, W

"""
2. Window Reverse: (Reconstructs the Original Tensor from Partitioned Windows)
--Parameters--
- @windows: torch.Tensor: Partitioned Windows of shape (num_windows*B, window_size, window_size, C)
- @window_size: int: Size of the Window
- @H: int: Original Height of the Tensor
- @W: int: Original Width of the Tensor
--Returns--
- torch.Tensor: Reconstructed Tensor of shape (B, H, W, C)

"""

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int)-> torch.Tensor:
  B = int(windows.shape[0] / (H * W / window_size / window_size))
  C = windows.shape[-1]

  x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
  x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
  
  return x