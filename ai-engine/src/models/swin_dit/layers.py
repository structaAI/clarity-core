"""
Layers for the SwinDiT backbone.

Modules
-------
CLIPProjection       — Projects CLIP embeddings into the model's hidden dim.
TimeStepEmbedding    — Sinusoidal time embedding with optional multi-signal
                       fusion (CLIP text, degradation type, severity).
SwinPatchEmbed       — Conv2d patch partition + LayerNorm serialisation.
"""

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# CLIP Projection
# ---------------------------------------------------------------------------

class CLIPProjection(nn.Module):
  """
  Maps CLIP output embeddings (768 or 1024-dim) into the model's hidden
  dimension so they can be fused with the time embedding before conditioning.

  Parameters
  ----------
  clip_dim : int
      Output dimension of the CLIP encoder (e.g. 768 for ViT-L/14).
  model_dim : int
      Hidden dimension of the SwinDiT backbone (embed_dim in config).
  """

  def __init__(self, clip_dim: int, model_dim: int) -> None:
    super().__init__()
    self.proj: nn.Sequential = nn.Sequential(
      nn.Linear(clip_dim, model_dim),
      nn.SiLU(),
      nn.Linear(model_dim, model_dim),
    )

  def forward(self, clip_embed: torch.Tensor) -> torch.Tensor:
    return self.proj(clip_embed)


# ---------------------------------------------------------------------------
# TimeStep Embedding
# ---------------------------------------------------------------------------

class TimeStepEmbedding(nn.Module):
  """
  Sinusoidal timestep embedding with optional multi-signal conditioning.

  The embedding is built from up to four signals fused via cross-attention:

  1. t_embed    — Sinusoidal clock, always present.
  2. clip_embed — CLIP text/image conditioning (optional).
  3. type_embed — Discrete degradation-type token (optional).
  4. sev_embed  — Scalar severity in [0, 1] (optional).

  When no optional signals are provided the module behaves identically to a
  standard sinusoidal timestep MLP (unconditional mode).

  Parameters
  ----------
  hidden_size : int
      Output embedding dimensionality.
  frequency_embedding_size : int
      Internal sinusoidal frequency count (default: 256).
  num_degradation_types : int
      Vocabulary size for the discrete degradation-type embedding (default: 5).
  clip_dim : int
      Input dimensionality of raw CLIP embeddings (default: 768).
  """

  def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, num_degradation_types: int = 5, clip_dim: int = 768) -> None:
    super().__init__()

    self.hidden_size = hidden_size
    self.frequency_embedding_size = frequency_embedding_size

    # 1. Sinusoidal → hidden MLP
    self.mlp: nn.Sequential = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size),
    )

    # 2. CLIP projection: raw CLIP dim → hidden_size
    #    Always constructed so the attribute always exists on the module.
    self.clip_proj: CLIPProjection = CLIPProjection(
      clip_dim=clip_dim,
      model_dim=hidden_size,
    )

    # 3. Discrete degradation-type embedding
    self.type_embeddings: nn.Embedding = nn.Embedding(
      num_degradation_types, hidden_size
    )

    # 4. Scalar severity encoder [0, 1] → hidden_size
    self.severity_encoder: nn.Sequential = nn.Sequential(
      nn.Linear(1, hidden_size // 4),
      nn.SiLU(),
      nn.Linear(hidden_size // 4, hidden_size),
    )

    # 5. Tripartite fusion via cross-attention
    self.fusion_attention: nn.MultiheadAttention = nn.MultiheadAttention(
      hidden_size, num_heads=8, batch_first=True
    )

  # ------------------------------------------------------------------
  # Sinusoidal helper
  # ------------------------------------------------------------------

  def _sinusoidal_embed(self, t: torch.Tensor) -> torch.Tensor:
    """
    Compute sinusoidal positional encoding for scalar timestep tensor t.

    Parameters
    ----------
    t : torch.Tensor
        Shape [] or [B] — integer timestep values.

    Returns
    -------
    torch.Tensor
        Shape [B, frequency_embedding_size].
    """
    half_dim = self.frequency_embedding_size // 2
    freqs = torch.exp(
      -math.log(10000)
      * torch.arange(0, half_dim, device=t.device).float()
      / half_dim
    )
    if t.ndim == 0:
      t = t.unsqueeze(0)
    args = t[:, None].float() * freqs[None, :]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

  # ------------------------------------------------------------------
  # Forward
  # ------------------------------------------------------------------

  def forward(self, t: torch.Tensor, clip_embeddings: Optional[torch.Tensor] = None, degradation_type: Optional[torch.Tensor] = None, severity: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Parameters
    ----------
    t : torch.Tensor
        Shape [B] — diffusion timestep indices.
    clip_embeddings : torch.Tensor, optional
        Shape [B, clip_dim] — raw CLIP encoder output.
    degradation_type : torch.Tensor, optional
        Shape [B] — integer degradation class indices.
    severity : torch.Tensor, optional
        Shape [B] — float severity values in [0, 1].

    Returns
    -------
    torch.Tensor
        Shape [B, hidden_size] — fused conditioning vector.
    """
    # 1. Base timestep embedding
    t_freq = self._sinusoidal_embed(t)
    t_embed = self.mlp(t_freq)              # [B, hidden_size]

    # 2. Accumulate token sequence for fusion attention
    tokens = [t_embed.unsqueeze(1)]         # [B, 1, hidden_size]

    if clip_embeddings is not None:
      c_embed = self.clip_proj(clip_embeddings)   # [B, hidden_size]
      tokens.append(c_embed.unsqueeze(1))         # [B, 1, hidden_size]

    if degradation_type is not None:
      type_embed = self.type_embeddings(degradation_type)  # [B, hidden_size]
      tokens.append(type_embed.unsqueeze(1))

    if severity is not None:
      sev_input = severity.float().unsqueeze(-1)           # [B, 1]
      sev_embed = self.severity_encoder(sev_input)         # [B, hidden_size]
      tokens.append(sev_embed.unsqueeze(1))

    # 3. Fuse: query = t_embed, keys/values = all tokens
    token_seq = torch.cat(tokens, dim=1)    # [B, T, hidden_size]
    query = t_embed.unsqueeze(1)            # [B, 1, hidden_size]

    fused, _ = self.fusion_attention(query, token_seq, token_seq)

    return fused.squeeze(1)                 # [B, hidden_size]


# ---------------------------------------------------------------------------
# Patch Embedding
# ---------------------------------------------------------------------------

class SwinPatchEmbed(nn.Module):
  """
  Partition the 2-D latent into non-overlapping patches, project each patch
  into an embedding vector, and serialise to a 1-D sequence.

  Parameters
  ----------
  patch_size : int
      Side length of each square patch (default: 2).
  no_of_in_channels : int
      Number of input channels (default: 4, matching the SDXL-VAE latent).
  embed_dim : int
      Projection embedding dimensionality (default: 768).

  Returns
  -------
  Tuple[torch.Tensor, Tuple[int, int]]
      - Normalised sequence: [B, H/P * W/P, embed_dim]
      - (H, W) grid dimensions after patch partition.
  """

  def __init__(self, patch_size: int = 2, no_of_in_channels: int = 4, embed_dim: int = 768) -> None:
    super().__init__()
    self.projection: nn.Conv2d = nn.Conv2d(
      no_of_in_channels,
      embed_dim,
      kernel_size=patch_size,
      stride=patch_size,
    )
    self.norm: nn.LayerNorm = nn.LayerNorm(embed_dim)

  def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    x = self.projection(x)           # [B, embed_dim, H/P, W/P]
    _, _, H, W = x.shape
    x = x.flatten(2).transpose(1, 2)  # [B, H*W, embed_dim]
    return self.norm(x), (H, W)