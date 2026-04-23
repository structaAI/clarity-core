"""
backbone.py — SwinDiT Backbone
================================
Changes from previous version
------------------------------
1. Final LayerNorm before the linear head.
   DiT-style models universally add a norm before the output projection.
   Without it, the final linear layer sees unnormalised activations whose
   scale drifts during training, causing instability and slow convergence
   especially at low LR (as in perceptual fine-tuning).

2. LR token extraction and threading.
   The input x has shape [B, 8, H, W] (8 = 4 HR channels cat 4 LR channels).
   After patch embedding, the full 8-channel sequence is used for self-attention
   conditioning via the time embedding, but the LR tokens are also extracted
   separately (by re-embedding only channels 4:8) and passed to the single
   mid-depth LRCrossAttention block. This gives the model an explicit,
   structured pathway to copy clean structure from LR into the denoised HR.

3. Mid-depth LR bridge injection.
   Exactly one SwinBlock is flagged is_lr_bridge=True at index len(blocks)//2.
   All other blocks are unchanged. This keeps parameter overhead small
   (~2% increase) while providing the cross-attention benefit.

4. out_channels fixed at 8 (unchanged from previous version).
   The model predicts noise for all 8 input channels; the training loop
   slices [:, :4] to supervise only the HR channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from .layers import TimeStepEmbedding, SwinPatchEmbed
from .blocks import SwinBlock


class SwinDiT(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        c = config.model

        self.out_channels = getattr(c, "out_channels", 8)

        # Primary patch embedding for the full 8-channel input
        self.patch_embedding = SwinPatchEmbed(
            patch_size=c.patch_size,
            no_of_in_channels=c.in_channels,    # 8 (HR+LR channels)
            embed_dim=c.embed_dim,
        )

        # Secondary patch embedding for LR-only channels (used by LRCrossAttention)
        # Channels 4:8 of the input are the upsampled LR latent.
        # We embed them independently at the same patch size to get clean LR tokens.
        self.lr_patch_embedding = SwinPatchEmbed(
            patch_size=c.patch_size,
            no_of_in_channels=c.in_channels // 2,   # 4 (LR channels only)
            embed_dim=c.embed_dim,
        )

        self.time_embedding = TimeStepEmbedding(hidden_size=c.embed_dim)

        # Build per-block head list from hierarchical stage config
        heads_per_block = []
        for stage_idx, depth in enumerate(c.depths):
            heads_per_block.extend([c.num_heads[stage_idx]] * depth)

        total_blocks = sum(c.depths)
        mid_idx      = total_blocks // 2   # index of the single LR cross-attention block

        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=c.embed_dim,
                num_heads=heads_per_block[i],
                window_size=c.window_size,
                shifted=(i % 2 != 0),
                use_pswa_bridge=c.use_pswa_bridge,
                bridge_type=getattr(c, "bridge_type", None),
                is_lr_bridge=(i == mid_idx),   # single LR cross-attention injection
            )
            for i in range(total_blocks)
        ])

        # Final norm (DiT standard) — stabilises the linear head input
        self.final_norm = nn.LayerNorm(c.embed_dim)

        self.final_layer = nn.Linear(
            c.embed_dim,
            c.patch_size ** 2 * self.out_channels,
        )

    def forward(
        self,
        x,
        t=None,
        clip_embeddings=None,
        degradation_type=None,
        severity=None,
        precomputed_cond=None,
    ) -> torch.Tensor:
        # Cast to model dtype
        dtype = self.patch_embedding.projection.weight.dtype
        x = x.to(dtype)

        # --- Extract LR channels before full embedding ---
        # x[:, 4:8] = upsampled LR latent (injected by training loop via cat)
        # We embed these separately to get clean LR tokens for cross-attention.
        if x.shape[1] == 8:
            lr_raw = x[:, 4:8].contiguous()   # [B, 4, H_px, W_px]
            lr_tokens, _ = self.lr_patch_embedding(lr_raw)   # [B, N, C]
        else:
            lr_tokens = None   # unconditional / inference without LR

        # --- Primary patch embedding (full 8-channel input) ---
        x, (H, W) = self.patch_embedding(x)   # [B, N, C]

        # --- Time / text conditioning ---
        if precomputed_cond is not None:
            t_feature = precomputed_cond.view(x.shape[0], -1)
        else:
            t_feature = self.time_embedding(
                t, clip_embeddings, degradation_type, severity
            )

        # --- Swin blocks ---
        for block in self.blocks:
            x = block(x, t_feature, H, W, lr_tokens=lr_tokens)

        # --- Final norm + linear head ---
        x = self.final_norm(x)
        x = self.final_layer(x)

        B, N, Cpatch = x.shape
        out_ch = self.out_channels
        p = int((Cpatch // out_ch) ** 0.5)

        return (
            x.view(B, H, W, p, p, out_ch)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .view(B, out_ch, H * p, W * p)
        )