import torch
import torch.nn as nn
from typing import Optional
from .layers import TimeStepEmbedding, SwinPatchEmbed
from .blocks import SwinBlock

class SwinDiT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        c = config.model
        self.patch_embedding = SwinPatchEmbed(
            patch_size=c.patch_size,
            no_of_in_channels=c.in_channels,
            embed_dim=c.embed_dim,
        )
        self.time_embedding = TimeStepEmbedding(hidden_size=c.embed_dim)
        
        heads_per_block = []
        for stage_idx, depth in enumerate(c.depths):
            heads_per_block.extend([c.num_heads[stage_idx]] * depth)

        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=c.embed_dim,
                num_heads=heads_per_block[i],  
                window_size=c.window_size,
                shifted=(i % 2 != 0),
                use_pswa_bridge=c.use_pswa_bridge,
                bridge_type=getattr(c, "bridge_type", None),          
            )
            for i in range(sum(c.depths))
        ])
        self.final_layer = nn.Linear(c.embed_dim, c.patch_size ** 2 * c.in_channels)

    def forward(self, x, t=None, clip_embeddings=None, degradation_type=None, severity=None, precomputed_cond=None):
        # 1. Patch partition
        x, (H, W) = self.patch_embedding(x) 

        # 2. Conditioning: Force [Batch, Embed_Dim] shape
        if precomputed_cond is not None:
            t_feature = precomputed_cond.view(x.shape[0], -1) 
        else:
            t_feature = self.time_embedding(t, clip_embeddings, degradation_type, severity)

        # 3. Swin blocks (Broadcasting t_feature [B, 1, D] happens inside blocks)
        for block in self.blocks:
            x = block(x, t_feature, H, W)

        # 4. Fold back
        x = self.final_layer(x)
        B, N, Cpatch = x.shape
        in_ch = self.patch_embedding.projection.in_channels
        p = int((Cpatch // in_ch) ** 0.5)
        return (x.view(B, H, W, p, p, in_ch).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, in_ch, H * p, W * p))