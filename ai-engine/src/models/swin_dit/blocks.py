# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Dependency Imports
from .utils import window_partition, window_reverse

"""
Pseudo-Shifted Window Attention(PSWA)

--Parameters--
- @dim: int: Dimension of Input Features

--Returns--
- torch.Tensor : The output is the Tensor with all HF (High-Frequency) Features from the Image Patches

Over here, we define a bridge module that extracts high-frequency features from the I/P patches
This has been done via a simple MLP with GELU activation function.
"""
class PSWABridge(nn.Module):
  def __init__(self, dim: int)-> None:
    super().__init__()

    # Defining the High-Frequency Feature Extractor MLP (GELU Activation Function)
    self.high_frequency_extractor: nn.Sequential = nn.Sequential(
      nn.Linear(dim, dim //4),
      nn.GELU(),
      nn.Linear(dim // 4, dim)
    )

    # Scaling Parameters and Layer Normalization for Stability since we are Reducing Dimensions
    self.scale: nn.Parameter = nn.Parameter(torch.ones(1) * 0.1)
    self.norm: nn.LayerNorm = nn.LayerNorm(dim)
  
  def forward(self, x: torch.Tensor)-> torch.Tensor:
    # Returning the Scaled High-Frequecy Features added to the Original Input Tensor
    return x + self.high_frequency_extractor(self.norm(x)) * self.scale


"""
Swin Transformer Block

--Parameters--
- @dim: int: Dimension of Input Features
- @num_heads: int: Number of Attention Heads
- @window_size: int: Size of the Shifting Window (default: 8)
- @shifted: bool: Flag variable --> Whether SWA (Shifting Window Attention) is being used (default: False)
- @use_pswa_bridge: bool: Flag variable --> Whether to use PSWA Bridge Module (default: False)

--Returns--
- torch.Tensor: Output Tensor after passing through Swin Block

Over here, we define a bridge module that extracts high-frequency features from the I/P patches:

1. Frequency Isolation: This module acts as a "High-Frequency Feature Extractor," specifically 
   targeting the textural details (edges, noise patterns, and fine grains) that are typically 
   smoothed out during the window partitioning process.
2. Dimensional Bottleneck: By passing features through a (dim -> dim//4) compression, 
   the MLP identifies the most salient high-order features before re-projecting 
   them back to the original manifold.
3. Scaled Residual Injection: The high-frequency components are scaled by a factor 
   of 0.1 to stabilize the diffusion reverse process, ensuring that the additive 
   details do not overwhelm the structural foundation established by the 
   attention layers.
"""
class SwinBlock(nn.Module):
  def __init__(self, dim: int, num_heads: int, window_size: int = 8, shifted: bool = False, use_pswa_bridge: bool = False)-> None:
    super().__init__()

    # Defining Parameters for instance in constructor
    self.dim: int = dim
    self.num_heads: int = num_heads
    self.window_size: int = window_size
    self.shifted = shifted

    # Attention Head
    self.attention: nn.MultiheadAttention = nn.MultiheadAttention(dim, num_heads=num_heads, batch_first=True)
    # Layer Normalization
    self.norm1: nn.LayerNorm = nn.LayerNorm(dim)
    # PSWA Bridge
    self.bridge: Optional[PSWABridge] = PSWABridge(dim) if use_pswa_bridge else None
  
  def forward(self, x: torch.Tensor, t_embed: torch.Tensor, H: int, W: int)-> torch.Tensor:
    # Adding Time Step Embeddings to input tensor
    res = x + t_embed.unsqueeze(1)
    # Performing Layer Normalization on the patch embeddings
    x_norm = self.norm1(res)

    B, L, C = x_norm.shape # Batch Size (B), Length (L), Number of Channels (C): L=HxW
    x_grid = x_norm.view(B, H, W, C)

    # Checking shifted Flag variable
    if self.shifted:
      # Shift size must be half that of the window size
      shift_size = self.window_size // 2
      # We then perform rolling in order to cover the entire image by shifting different patches.
      x_grid = torch.roll(x_grid, shifts=(-shift_size, -shift_size), dims=(1, 2))

    x_windows, _, _= window_partition(x_grid, self.window_size) # Tensor, partition height (p_H), partition width (p_W)
    x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

    attention_output, _ = self.attention(x_windows, x_windows, x_windows) # Attention Output

    attention_output = attention_output.view(-1, self.window_size, self.window_size, C)
    x = window_reverse(attention_output, self.window_size, H, W)

    # Rechecking shifted Flag Variables
    if self.shifted:
      shift_size = self.window_size // 2
      x = torch.roll(x, shifts=(shift_size, shift_size), dims=(1, 2))

    # Convert 2D grid block to a 1D sequence
    x = x.view(-1, L, C)

    # REsidual Connection
    x = res + x

    # Checking the bridge flag variable
    if self.bridge is not None:
      x = self.bridge(x)

    return x