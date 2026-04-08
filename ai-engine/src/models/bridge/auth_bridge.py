"""
AuthBridge — Gated MLP conditioning bridge.

Maps a conditioning vector (e.g. time embedding) into the feature space via a
gated projection. The gate is sigmoid-activated and zero-initialised, so the
bridge starts as identity and only activates as training converges.

Architecture:
    x_out = x + gate(cond) * proj(x)

Parameters
----------
input_dim : int
    Dimensionality of the feature tensor x (sequence tokens).
output_dim : int
    Output dimensionality (must equal input_dim for residual compatibility).
cond_dim : int, optional
    Dimensionality of the conditioning vector. Defaults to output_dim.
"""

import torch
import torch.nn as nn
from typing import Optional


class AuthBridge(nn.Module):
  def __init__(
    self,
    input_dim: int,
    output_dim: int,
    cond_dim: Optional[int] = None,
  ) -> None:
    super().__init__()

    if input_dim != output_dim:
      raise ValueError(
        f"AuthBridge requires input_dim == output_dim for residual "
        f"compatibility, got {input_dim} vs {output_dim}."
      )

    cond_dim = cond_dim or output_dim

    # Feature projection: maps token features to output space.
    self.proj = nn.Sequential(
      nn.LayerNorm(input_dim),
      nn.Linear(input_dim, output_dim),
      nn.GELU(),
      nn.Linear(output_dim, output_dim),
    )

    # Gate: scalar per token, conditioned on the external signal.
    # Linear → sigmoid. Zero-init weight so gate starts fully closed.
    self.gate_fc = nn.Linear(cond_dim, output_dim)
    nn.init.zeros_(self.gate_fc.weight)
    nn.init.zeros_(self.gate_fc.bias)

    self.time_mlp = nn.Sequential(
      nn.Linear(1, output_dim),  # Scalar → vector
      nn.SiLU(),
      nn.Linear(output_dim, output_dim),
    ) if cond_dim == 1 else None

  def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Parameters
    ----------
    x : torch.Tensor
        Shape [B, N, D] — sequence of token features.
    cond : torch.Tensor, optional
        Shape [B, cond_dim] — conditioning vector (e.g. time embedding).
        If None, the gate defaults to 0.5 (half-open).

    Returns
    -------
    torch.Tensor
        Shape [B, N, D] — refined features with residual connection.
    """
    projected = self.proj(x)  # [B, N, D]

    if cond is not None:
      if self.time_mlp is not None:
        cond = self.time_mlp(cond.unsqueeze(-1))  # [B] → [B, 1] → [B, D]
      gate = torch.sigmoid(self.gate_fc(cond)).unsqueeze(1)
    else:
        gate = 0.5

    return x + gate * projected