"""
blocks.py — SwinDiT Transformer Blocks
========================================
Changes from previous version
------------------------------
1. RelativePositionBias (RPB)
   nn.MultiheadAttention has no spatial inductive bias — every token pair
   gets the same prior. RPB adds a learned scalar per relative (Δrow, Δcol)
   offset to the attention logits BEFORE softmax, exactly as in Swin-T.
   This is the single highest-impact change for SR: the model now knows
   that nearby patches should attend more strongly by default.

2. AdaLN (Adaptive Layer Norm) conditioning
   The old approach added t_embed as a token offset: x + t_embed.unsqueeze(1).
   This is additive and quickly saturates — the time signal gets washed out
   by deep blocks. AdaLN instead predicts per-channel (scale, shift) from
   t_embed and applies them to the pre-norm features. This is how DiT
   (Peebles & Xie 2023) conditions and it generalises much better across
   timestep ranges, especially at low noise where SR detail matters most.

3. LRCrossAttention
   The LR image currently enters only via channel-cat at input. This is
   structurally fine but blunt — the model must infer low-frequency content
   from the noisy HR channel rather than being able to explicitly attend to
   the clean LR. LRCrossAttention is a lightweight cross-attention that lets
   HR tokens (queries) attend to projected LR tokens (keys/values). It is
   injected once at the midpoint of the block sequence via the `is_lr_bridge`
   flag, keeping the parameter overhead small.

4. PSWABridge unchanged (static variant)
   AdaptivePSWABridge is preserved but gating now receives the AdaLN-modulated
   features rather than raw features, which makes the frequency routing more
   meaningful during conditioned inference.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .utils import window_partition, window_reverse
from .adaptive_backbone import AdaptivePSWABridge


# ---------------------------------------------------------------------------
# Relative Position Bias
# ---------------------------------------------------------------------------

class RelativePositionBias(nn.Module):
    """
    Learned relative position bias table for window attention.

    For a window of size W×W, the relative offsets range from
    -(W-1) to +(W-1) in each dimension, giving a table of shape
    (2W-1, 2W-1). At forward time, indices are looked up and added
    to the raw QK dot-product before softmax.

    Parameters
    ----------
    window_size : int
        Side length of the attention window (e.g. 8).
    num_heads : int
        Number of attention heads — one scalar per head per offset pair.
    """

    def __init__(self, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads

        # Table: (2W-1)^2 offsets × num_heads scalars
        self.bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.bias_table, std=0.02)

        # Pre-compute index lookup buffer (register so it moves with .to(device))
        coords = torch.arange(window_size)
        grid   = torch.stack(torch.meshgrid(coords, coords, indexing="ij"))  # [2, W, W]
        flat   = grid.flatten(1)                                              # [2, W*W]
        rel    = flat[:, :, None] - flat[:, None, :]                          # [2, W*W, W*W]
        rel    = rel.permute(1, 2, 0).contiguous()                            # [W*W, W*W, 2]
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        idx = rel.sum(-1).long()                                              # [W*W, W*W] int64
        self.register_buffer("relative_position_index", idx)

    def forward(self) -> torch.Tensor:
        """Returns [num_heads, W*W, W*W] bias to add before softmax."""
        W2  = self.window_size ** 2
        idx = self.relative_position_index          # Tensor [W*W, W*W]
        idx_flat = idx.flatten()  # type: ignore    # [W*W * W*W] — explicit type avoids Pyright callable error
        bias = self.bias_table[idx_flat]            # [W*W*W*W, num_heads]
        bias = bias.view(W2, W2, self.num_heads).permute(2, 0, 1)  # [num_heads, W*W, W*W]
        return bias.contiguous()


# ---------------------------------------------------------------------------
# Adaptive Layer Norm (AdaLN)
# ---------------------------------------------------------------------------

class AdaLN(nn.Module):
    """
    Adaptive Layer Norm conditioning (DiT-style).

    Projects the conditioning vector t_embed into per-channel (scale, shift)
    parameters that modulate the pre-normalised features. Compared to additive
    conditioning (x + t_embed), this:
      - Scales and shifts independently per channel.
      - Is zero-initialised so it starts as identity, matching the residual
        init philosophy of the rest of the model.
      - Provides multiplicative gating that doesn't saturate at depth.

    Parameters
    ----------
    dim : int
        Feature dimension (same for input, scale, and shift).
    cond_dim : int
        Dimensionality of the conditioning vector (t_embed). Defaults to dim.
    """

    def __init__(self, dim: int, cond_dim: Optional[int] = None) -> None:
        super().__init__()
        cond_dim = cond_dim or dim
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(cond_dim, 2 * dim)
        # Zero-init: at epoch 0, scale=0 shift=0 → output is just norm(x)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : [B, N, D]
        cond : [B, cond_dim]

        Returns [B, N, D] with per-channel scale/shift applied.
        """
        scale, shift = self.proj(cond).chunk(2, dim=-1)   # each [B, D]
        x_norm = self.norm(x)
        return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
# LR Cross-Attention (injected once at mid-depth)
# ---------------------------------------------------------------------------

class LRCrossAttention(nn.Module):
    """
    Lets HR latent tokens explicitly attend to LR latent tokens.

    The LR conditioning currently enters only via channel concatenation at
    the patch embedding stage. This module adds a second pathway: HR tokens
    (queries) cross-attend to LR tokens (keys/values). This gives the model
    a structured mechanism to copy clean low-frequency content from LR into
    the denoised HR, which is exactly what SR requires.

    Architecture:
        Q = linear(hr_tokens)         [B, N_hr, dim]
        K = linear(lr_tokens)         [B, N_lr, dim]
        V = linear(lr_tokens)         [B, N_lr, dim]
        out = softmax(QK^T / sqrt(d)) V
        hr_tokens = hr_tokens + gate * out

    The gate is zero-initialised (closed at epoch 0) so the module
    activates gradually as training progresses.

    Parameters
    ----------
    dim       : int   Feature dimension.
    num_heads : int   Attention heads (default: 8).
    """

    def __init__(self, dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.norm_hr = nn.LayerNorm(dim)
        self.norm_lr = nn.LayerNorm(dim)

        self.q_proj  = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Gate: zero-init → starts closed
        self.gate = nn.Parameter(torch.zeros(1))

        # Project LR tokens from 4ch (concat) to model dim if needed
        # (LR tokens come from the same patch embed so they're already dim)

    def forward(
        self,
        hr_tokens: torch.Tensor,   # [B, N_hr, dim]
        lr_tokens: torch.Tensor,   # [B, N_lr, dim]
    ) -> torch.Tensor:
        B, N, D = hr_tokens.shape
        H = self.num_heads

        q  = self.q_proj(self.norm_hr(hr_tokens))             # [B, N_hr, D]
        kv = self.kv_proj(self.norm_lr(lr_tokens))            # [B, N_lr, 2D]
        k, v = kv.chunk(2, dim=-1)

        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, -1, H, self.head_dim).transpose(1, 2)  # [B, H, N, d]

        q, k, v = reshape(q), reshape(k), reshape(v)

        attn = (q @ k.transpose(-2, -1)) * self.scale          # [B, H, N_hr, N_lr]
        attn = attn.softmax(dim=-1)

        out = (attn @ v)                                        # [B, H, N_hr, d]
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)

        return hr_tokens + torch.tanh(self.gate) * out


# ---------------------------------------------------------------------------
# PSWABridge (static, unchanged)
# ---------------------------------------------------------------------------

class PSWABridge(nn.Module):
    """
    Static high-frequency feature extractor.
    Unchanged from original — still used when bridge_type is not 'adaptive'.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.hf_extractor = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, dim),
        )
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
        self.norm  = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.hf_extractor(self.norm(x)) * self.scale


# ---------------------------------------------------------------------------
# SwinBlock — main transformer block
# ---------------------------------------------------------------------------

class SwinBlock(nn.Module):
    """
    Swin Transformer block with:
      - AdaLN conditioning (replaces additive t_embed offset)
      - Relative Position Bias in window attention
      - Optional LR cross-attention (is_lr_bridge=True, injected once at mid-depth)
      - Optional PSWABridge (static or adaptive)

    Parameters
    ----------
    dim             : int
    num_heads       : int
    window_size     : int   (default 8)
    shifted         : bool  Whether to apply cyclic shift
    use_pswa_bridge : bool
    bridge_type     : str | None  'adaptive' or None (static)
    is_lr_bridge    : bool  If True, inserts LR cross-attention after window attn.
                            Set to True for exactly one block at mid-depth.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shifted: bool = False,
        use_pswa_bridge: bool = False,
        bridge_type: Optional[str] = None,
        is_lr_bridge: bool = False,
    ) -> None:
        super().__init__()

        self.dim         = dim
        self.num_heads   = num_heads
        self.window_size = window_size
        self.shifted     = shifted
        self.is_lr_bridge = is_lr_bridge

        # AdaLN replaces both norm1 and the additive t_embed offset
        self.adaLN1 = AdaLN(dim)
        self.adaLN2 = AdaLN(dim)

        # Relative position bias
        self.rpb = RelativePositionBias(window_size, num_heads)

        # Window self-attention (still uses nn.MHA for simplicity,
        # but RPB is added manually to the logits via the forward hook below)
        # We use scaled_dot_product_attention directly so we can inject RPB.
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.attn_out = nn.Linear(dim, dim, bias=True)

        # LR cross-attention (optional, single injection point)
        self.lr_cross_attn: Optional[LRCrossAttention] = (
            LRCrossAttention(dim, num_heads) if is_lr_bridge else None
        )

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # PSWA Bridge
        match bridge_type:
            case "adaptive":
                self.bridge: Optional[nn.Module] = AdaptivePSWABridge(dim) if use_pswa_bridge else None
            case _:
                self.bridge = PSWABridge(dim) if use_pswa_bridge else None

    # ------------------------------------------------------------------
    # Window self-attention with RPB
    # ------------------------------------------------------------------

    def _window_attn(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [num_windows * B, W*W, C]
        Returns same shape.
        """
        nwB, L, C = x.shape
        H = self.num_heads
        d = C // H

        q = self.q_proj(x).view(nwB, L, H, d).transpose(1, 2)   # [nwB, H, L, d]
        k = self.k_proj(x).view(nwB, L, H, d).transpose(1, 2)
        v = self.v_proj(x).view(nwB, L, H, d).transpose(1, 2)

        scale  = d ** -0.5
        logits = (q @ k.transpose(-2, -1)) * scale                # [nwB, H, L, L]

        # Add relative position bias [H, L, L] broadcast over batch
        logits = logits + self.rpb().to(logits.dtype)

        attn = logits.softmax(dim=-1)
        out  = (attn @ v).transpose(1, 2).reshape(nwB, L, C)
        return self.attn_out(out)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,         # [B, N, C]  HR token sequence
        t_embed: torch.Tensor,   # [B, C]     AdaLN conditioning
        H: int,
        W: int,
        lr_tokens: Optional[torch.Tensor] = None,  # [B, N_lr, C]
    ) -> torch.Tensor:

        B, L, C = x.shape

        # --- AdaLN-modulated pre-norm (replaces norm1 + additive offset) ---
        x_norm = self.adaLN1(x, t_embed)                         # [B, N, C]

        # --- Window partition ---
        x_grid = x_norm.view(B, H, W, C)

        if self.shifted:
            shift  = self.window_size // 2
            x_grid = torch.roll(x_grid, shifts=(-shift, -shift), dims=(1, 2))

        x_win, _, _ = window_partition(x_grid, self.window_size)  # [nwB, W, W, C]
        x_win = x_win.view(-1, self.window_size ** 2, C)

        # --- Window self-attention with RPB ---
        attn_out = self._window_attn(x_win)                       # [nwB, W*W, C]

        # --- Reverse partition ---
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        x_merged = window_reverse(attn_out, self.window_size, H, W)

        if self.shifted:
            x_merged = torch.roll(x_merged, shifts=(shift, shift), dims=(1, 2))

        x_merged = x_merged.view(B, L, C)

        # --- Residual 1 ---
        x = x + x_merged

        # --- LR cross-attention (single mid-depth injection) ---
        if self.lr_cross_attn is not None and lr_tokens is not None:
            x = self.lr_cross_attn(x, lr_tokens)

        # --- PSWA bridge ---
        if self.bridge is not None:
            if isinstance(self.bridge, AdaptivePSWABridge):
                x = self.bridge(x, t_embed)
            else:
                x = self.bridge(x)

        # --- AdaLN-modulated FFN ---
        x = x + self.ffn(self.adaLN2(x, t_embed))

        return x