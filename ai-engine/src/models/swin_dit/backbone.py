import torch
import torch.nn as nn
from typing import Optional

from .layers import TimeStepEmbedding, SwinPatchEmbed
from .blocks import SwinBlock

class SwinDiT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        c = config.model

        self.patch_embedding: SwinPatchEmbed = SwinPatchEmbed(
            patch_size=c.patch_size,
            no_of_in_channels=c.in_channels,
            embed_dim=c.embed_dim,
        )

        self.time_embedding: TimeStepEmbedding = TimeStepEmbedding(
            hidden_size=c.embed_dim
        )

        heads_per_block = []
        for stage_idx, depth in enumerate(c.depths):
            heads_per_block.extend([c.num_heads[stage_idx]] * depth)

        bridge_type = getattr(c, "bridge_type", None)

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
        precomputed_cond: Optional[torch.Tensor] = None, # Added for AuthBridge compatibility
    ) -> torch.Tensor:
        # 1. Patch partition
        x, (H, W) = self.patch_embedding(x) 

        # 2. Conditioning: Use AuthBridge output if provided, else fallback
        if precomputed_cond is not None:
            t_feature = precomputed_cond
        else:
            t_feature = self.time_embedding(
                t,
                clip_embeddings=clip_embeddings,
                degradation_type=degradation_type,
                severity=severity,
            )

        # 3. Swin blocks
        for block in self.blocks:
            x = block(x, t_feature, H, W)

        # 4. Project and Fold
        x = self.final_layer(x)
        B, _, Cpatch = x.shape
        in_channel = self.patch_embedding.projection.in_channels
        p = int((Cpatch // in_channel) ** 0.5)

        return (
            x.view(B, H, W, p, p, in_channel)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(B, in_channel, H * p, W * p)
        )