# Imports
import torch
import torch.nn as nn

# Dependency Imports
from .layers import TimeStepEmbedding, SwinPatchEmbed
from .blocks import SwinBlock

# Defining the SwinDiT BackBone Model
"""
SwinDiT Backbone Model

- Input: Image Tensor and Time Step Tensor
- Output: Reconstructed Image Tensor

----Parameters----
- @config: Object: YAML Configuration containing Model Hyperparameters

---Architecture Overview---
1. Patch Partition Layer: Convert Input Image into Patch Embeddings
2. Time Step Embedding: Generate Time Step Features

These 2 together form the Patch Embedding Module
"""
class SwinDiT(nn.Module):
  def __init__(self, config)-> None:
    super().__init__()

    # Extracting Model Configuration
    c = config.model

    # Defining Patch Embedding Layer
    self.patch_embedding: SwinPatchEmbed = SwinPatchEmbed(
      patch_size=c.patch_size,
      no_of_in_channels=c.in_channels,
      embed_dim=c.embed_dim
    )

    # Defining Time Step Embedding Layer
    self.time_embedding: TimeStepEmbedding = TimeStepEmbedding(hidden_size=c.embed_dim)

    # Defining Swin Transformer Blocks
    self.blocks: nn.ModuleList = nn.ModuleList([
      SwinBlock(
        dim=c.embed_dim,
        num_heads=c.num_heads[0],
        window_size=c.window_size,
        shifted=(i % 2 != 0),
        use_pswa_bridge=c.use_pswa_bridge
      ) for i in range(sum(c.depths))
    ])

    # Final Linear Layer to map back to Image Space
    self.final_layer: nn.Linear = nn.Linear(c.embed_dim, c.patch_size** 2 * c.in_channels)
  
  # Forward Pass through the SwinDiT Model
  def forward(self, x: torch.Tensor, t: torch.Tensor)-> torch.Tensor:

    x , (H, W)= self.patch_embedding(x) # Dimmension: 2 --> torch.Tensor, Tuple[int, int]
    t_feature = self.time_embedding(t) # Time Step Features

    # Traversing through Swin Transformer Blocks
    for block in self.blocks:
      x = block(x, t_feature, H, W)
    
    x = self.final_layer(x)

    B, _, C = x.shape # Batch Size(B), Number of Patches(N), Channels(C)

    # Calculating Patch Size
    in_channel = self.patch_embedding.projection.in_channels
    p = int((C//in_channel)**0.5)
    
    # Reshaping back to Image Space
    return x.view(B, H, W, p, p, in_channel).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, in_channel, H*p, W*p)