# 

import torch
import numpy as np
from typing import Optional, Literal

class GaussianDiffusion:
  def __init__(self, num_timesteps: int = 1000, schedule: Literal['linear', 'cosine'] = 'linear') -> None:
    self.num_timesteps = num_timesteps
    self.schedule = schedule

    if self.schedule == 'linear':
      self.betas = torch.linspace(1e-4, 0.02, num_timesteps, dtype=torch.float32)
    elif self.schedule == 'cosine':
      self.betas = self._cosine_beta_schedule(num_timesteps)
    else:
      raise ValueError(f"Unverified Schedule Type: {self.schedule}")
    
    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

  def _cosine_beta_schedule(self, num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1.0 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

  def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    if noise is None:
      noise = torch.randn_like(x_0)
    
    sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

  def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    batch_size = t.shape[0]
    out = a.to(t.device).gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))