import torch
import torch.nn as nn

from .layers import TimeStepEmbedding, SwinPatchEmbed
from .blocks import SwinBlock

class SwinDiT(nn.Module):
  def __init__(self, config)-> None:
    super().__init__()

    c = config.model

    self.patch_embedding: SwinPatchEmbed = SwinPatchEmbed(
      patch_size=c.patch_size,
      no_of_in_channels=c.in_channels,
      embed_dim=c.embed_dim
    )

    self.time_embedding: TimeStepEmbedding = TimeStepEmbedding(hidden_size=c.embed_dim)

    self.blocks: nn.ModuleList = nn.ModuleList([
      SwinBlock(
        dim=c.embed_dim,
        num_heads=c.num_heads[0],
        window_size=c.window_size,
        shifted=(i % 2 != 0),
        use_pswa_bridge=c.use_pswa_bridge
      ) for i in range(sum(c.depths))
    ])

    self.final_layer: nn.Linear = nn.Linear(c.embed_dim, c.patch_size** 2 * c.in_channels)
  
  def forward(self, x: torch.Tensor, t: torch.Tensor)-> torch.Tensor:
    x , (H, W)= self.patch_embedding(x)
    t_feature = self.time_embedding(t)

    for block in self.blocks:
      x = block(x, t_feature, H, W)
    
    x = self.final_layer(x)

    B, N, C = x.shape
    in_channel = self.patch_embedding.projection.in_channels
    p = int((C//in_channel)**0.5)
    
    return x.view(B, H, W, p, p, in_channel).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, in_channel, H*p, W*p)