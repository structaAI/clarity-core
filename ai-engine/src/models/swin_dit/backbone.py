"""
SwinDiT Backbone.

The backbone accepts an optional conditioning bundle so the same model
supports all four inference modes without architectural changes:

  Mode A — Unconditional denoising    : forward(x, t)
  Mode B — Type/severity denoising    : forward(x, t, degradation_type=..., severity=...)
  Mode C — Text-guided SR             : forward(x, t, clip_embeddings=...)
  Mode D — Fully conditioned SR       : forward(x, t, clip_embeddings=...,
                                               degradation_type=..., severity=...)

Architecture Overview
---------------------
1. SwinPatchEmbed   — Partition latent into patch tokens.
2. TimeStepEmbedding— Build a conditioning vector from (t, CLIP, type, severity).
3. SwinBlock stack  — Hierarchical window-attention blocks.
4. Linear head      — Project back to pixel space per patch.
5. Pixel-shuffle    — Fold patches into the reconstructed latent.
"""

import torch
import torch.nn as nn
from typing import Optional

from .layers import TimeStepEmbedding, SwinPatchEmbed
from .blocks import SwinBlock


class SwinDiT(nn.Module):
  def __init__(self, config) -> None:
    super().__init__()

    c = config.model

    # 1. Patch partition
    self.patch_embedding: SwinPatchEmbed = SwinPatchEmbed(
      patch_size=c.patch_size,
      no_of_in_channels=c.in_channels,
      embed_dim=c.embed_dim,
    )

    # 2. Timestep + conditioning embedding
    self.time_embedding: TimeStepEmbedding = TimeStepEmbedding(
        hidden_size=c.embed_dim
    )

    # FIX: build per-stage head list matching the depths config so each
    # block gets the correct number of attention heads for its stage.
    # Previously ALL blocks used c.num_heads[0] (always 3 heads).
    #
    # Example: depths=[2,2,6,2], num_heads=[3,6,12,24]
    #   → blocks 0-1  use 3  heads  (stage 0, depth 2)
    #   → blocks 2-3  use 6  heads  (stage 1, depth 2)
    #   → blocks 4-9  use 12 heads  (stage 2, depth 6)
    #   → blocks 10-11 use 24 heads (stage 3, depth 2)
    heads_per_block = []
    for stage_idx, depth in enumerate(c.depths):
      heads_per_block.extend([c.num_heads[stage_idx]] * depth)

    bridge_type = getattr(c, "bridge_type", None)   # e.g. "adaptive" or None

    # 3. Swin blocks — alternating regular / shifted windows
    self.blocks: nn.ModuleList = nn.ModuleList([
      SwinBlock(
        dim=c.embed_dim,
        num_heads=heads_per_block[i],  
        window_size=c.window_size,
        shifted=(i % 2 != 0),
        use_pswa_bridge=c.use_pswa_bridge,
        bridge_type=bridge_type,          
      )
      for i in range(sum(c.depths))
    ])

    # 4. Output projection: tokens → flattened patch pixels
    self.final_layer: nn.Linear = nn.Linear(
        c.embed_dim, c.patch_size ** 2 * c.in_channels
    )

  def forward(
      self,
      x: torch.Tensor,
      t: Optional[torch.Tensor] = None,
      clip_embeddings: Optional[torch.Tensor] = None,
      degradation_type: Optional[torch.Tensor] = None,
      severity: Optional[torch.Tensor] = None,
  ) -> torch.Tensor:
    """
    Parameters
    ----------
    x : torch.Tensor
        Noisy latent, shape [B, C, H, W].
    t : torch.Tensor, optional
        Timestep indices, shape [B].
    clip_embeddings : torch.Tensor, optional
        CLIP text/image conditioning, shape [B, clip_dim].
    degradation_type : torch.Tensor, optional
        Integer degradation class, shape [B].
    severity : torch.Tensor, optional
        Float severity in [0, 1], shape [B].

    Returns
    -------
    torch.Tensor
        Reconstructed latent, same shape as x: [B, C, H, W].
    """
    # Patch embedding
    x, (H, W) = self.patch_embedding(x)    # [B, N, D], N = H*W

    # Conditioning vector — passes all optional signals through
    t_feature = self.time_embedding(
      t,
      clip_embeddings=clip_embeddings,
      degradation_type=degradation_type,
      severity=severity,
    )  # [B, embed_dim]

    # Swin block stack
    for block in self.blocks:
      x = block(x, t_feature, H, W)      # [B, N, D]

    # Project to pixel space
    x = self.final_layer(x)                # [B, N, patch_size² * C]

    B, _, Cpatch = x.shape
    in_channel = self.patch_embedding.projection.in_channels
    p = int((Cpatch // in_channel) ** 0.5)

    # Fold back to [B, C, H*p, W*p]
    return (
      x.view(B, H, W, p, p, in_channel)
      .permute(0, 5, 1, 3, 2, 4)
      .contiguous()
      .view(B, in_channel, H * p, W * p)
    )