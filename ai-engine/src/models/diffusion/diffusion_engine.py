"""
GaussianDiffusion — Forward and Reverse Diffusion Scheduler.

Supports:
  - Forward process : q_sample   (noising)
  - Reverse process : p_sample   (single DDPM step)
  - Reverse loop    : p_sample_loop (full T→0 trajectory)

Both unconditional denoising and LR-seeded super-resolution are supported
through the ``x_lr`` parameter in ``p_sample_loop``.

Parameters
----------
num_timesteps : int
    Total diffusion steps T (default: 1000).
schedule : {'linear', 'cosine'}
    Beta schedule type (default: 'linear').
"""

import torch
import numpy as np
from typing import Callable, Literal, Optional


class GaussianDiffusion:
  def __init__(self, num_timesteps: int = 1000, schedule: Literal["linear", "cosine"] = "linear") -> None:
    self.num_timesteps = num_timesteps
    self.schedule = schedule

    if schedule == "linear":
      self.betas = torch.linspace(1e-4, 0.02, num_timesteps, dtype=torch.float32)
    elif schedule == "cosine":
      self.betas = self._cosine_beta_schedule(num_timesteps)
    else:
      raise ValueError(f"Unsupported schedule: '{schedule}'. Choose 'linear' or 'cosine'.")

    self.alphas = 1.0 - self.betas
    self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    self.alphas_cumprod_prev = torch.cat([torch.ones(1, dtype=torch.float32), self.alphas_cumprod[:-1]])

    # Forward process coefficients
    self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
    self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    # Reverse process coefficients
    self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    # Posterior variance q(x_{t-1} | x_t, x_0)
    self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
    # Clip to avoid log(0) at t=0
    self.posterior_log_variance_clipped = torch.log(torch.clamp(self.posterior_variance, min=1e-20))
    self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
    self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

  # ------------------------------------------------------------------
  # Beta schedule
  # ------------------------------------------------------------------

  def _cosine_beta_schedule(self, num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = (torch.cos(((x / num_timesteps) + s) / (1.0 + s) * np.pi * 0.5) ** 2)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

  # ------------------------------------------------------------------
  # Shared helper
  # ------------------------------------------------------------------

  def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    batch_size = t.shape[0]
    out = a.to(t.device).gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

  # ------------------------------------------------------------------
  # Forward process: q(x_t | x_0)
  # ------------------------------------------------------------------

  def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Sample from the forward diffusion kernel q(x_t | x_0).

    Parameters
    ----------
    x_0 : torch.Tensor
        Clean latent, shape [B, C, H, W].
    t : torch.Tensor
        Timestep indices, shape [B].
    noise : torch.Tensor, optional
        Pre-sampled Gaussian noise.  Drawn fresh if None.

    Returns
    -------
    torch.Tensor
        Noisy latent x_t, shape [B, C, H, W].
    """
    if noise is None:
      noise = torch.randn_like(x_0)

    sqrt_alpha_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    return sqrt_alpha_t * x_0 + sqrt_one_minus_t * noise

  # ------------------------------------------------------------------
  # Reverse process: p(x_{t-1} | x_t)
  # ------------------------------------------------------------------

  def _predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    sqrt_recip_alpha_t = self._extract(1.0 / self.sqrt_alphas_cumprod, t, x_t.shape)
    sqrt_recip_m1 = self._extract(torch.sqrt(1.0 / self.alphas_cumprod - 1.0), t, x_t.shape)
    return sqrt_recip_alpha_t * x_t - sqrt_recip_m1 * eps

  def p_mean_variance(self, model_fn: Callable, x_t: torch.Tensor,t: torch.Tensor) -> dict:
    """
    Compute the posterior mean and variance for one reverse step.

    Parameters
    ----------
    model_fn : Callable
        Function (x_t, t) → predicted noise ε, same shape as x_t.
    x_t : torch.Tensor
        Noisy latent at step t, shape [B, C, H, W].
    t : torch.Tensor
        Current timestep indices, shape [B].

    Returns
    -------
    dict with keys: 'mean', 'variance', 'log_variance', 'pred_x0'.
    """
    # Predict noise from model
    eps_pred = model_fn(x_t, t)

    # Estimate clean x_0
    pred_x0 = self._predict_x0_from_eps(x_t, t, eps_pred)
    # Clip to stable range
    pred_x0 = pred_x0.clamp(-1.0, 1.0)

    # Posterior mean: μ_θ(x_t, t)
    coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
    coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
    mean = coef1 * pred_x0 + coef2 * x_t

    variance = self._extract(self.posterior_variance, t, x_t.shape)
    log_variance = self._extract(self.posterior_log_variance_clipped, t, x_t.shape)

    return {
      "mean": mean,
      "variance": variance,
      "log_variance": log_variance,
      "pred_x0": pred_x0,
    }

  @torch.no_grad()
  def p_sample(self, model_fn: Callable, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Perform a single DDPM reverse-diffusion step.

    Parameters
    ----------
    model_fn : Callable
        Function (x_t, t) → predicted ε.
    x_t : torch.Tensor
        Noisy latent at step t, shape [B, C, H, W].
    t : torch.Tensor
        Current timestep indices, shape [B].

    Returns
    -------
    torch.Tensor
        Denoised latent x_{t-1}, shape [B, C, H, W].
    """
    out = self.p_mean_variance(model_fn, x_t, t)

    noise = torch.randn_like(x_t)
    # No noise at t=0 (final step)
    nonzero_mask = (t != 0).float().reshape(-1, *((1,) * (x_t.ndim - 1)))

    return out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise

  @torch.no_grad()
  def p_sample_loop(self, model_fn: Callable, shape: tuple, device: torch.device, x_lr: Optional[torch.Tensor] = None, num_inference_steps: Optional[int] = None, progress_callback: Optional[Callable[[int, torch.Tensor], None]] = None) -> torch.Tensor:
    """
    Full reverse diffusion loop: sample x_0 from x_T.

    Supports two modes:

    Unconditional denoising
        ``x_lr=None`` — starts from pure Gaussian noise.

    Super-resolution
        ``x_lr`` is a low-resolution latent — blends LR content with noise
        at the starting timestep so the reverse process refines it.

    Parameters
    ----------
    model_fn : Callable
        Function (x_t, t) → predicted ε.  All external conditioning
        (CLIP embeddings, degradation type, severity) should be captured
        inside this closure before passing it here.
    shape : tuple
        Output tensor shape (B, C, H, W).
    device : torch.device
        Device to run sampling on.
    x_lr : torch.Tensor, optional
        LR latent for seeded SR initialisation, shape (B, C, H, W).
    num_inference_steps : int, optional
        Number of reverse steps to run.  Defaults to ``num_timesteps``
        (full schedule).  Values < num_timesteps evenly subsample the
        timestep sequence for faster inference.
    progress_callback : Callable[[int, Tensor], None], optional
        Called after each step with (step_index, current_x).  Useful for
        tqdm progress bars or intermediate visualisation.

    Returns
    -------
    torch.Tensor
        Denoised latent x_0, shape (B, C, H, W).
    """
    B = shape[0]

    # Build the subset of timesteps to use
    if num_inference_steps is None or num_inference_steps >= self.num_timesteps:
      timesteps = list(reversed(range(self.num_timesteps)))
    else:
      # Evenly spaced subset (stride > 1)
      stride = self.num_timesteps // num_inference_steps
      timesteps = list(reversed(range(0, self.num_timesteps, stride)))

    # Initial latent: pure noise or LR-seeded
    if x_lr is not None:
      # Seed from LR latent noised to the first scheduled timestep
      t_start = torch.tensor([timesteps[0]] * B, device=device)
      x = self.q_sample(x_lr.to(device), t_start, noise=torch.randn(shape, device=device))
    else:
      x = torch.randn(shape, device=device)

    for step_idx, t_val in enumerate(timesteps):
      t = torch.full((B,), t_val, device=device, dtype=torch.long)
      x = self.p_sample(model_fn, x, t)

      if progress_callback is not None:
        progress_callback(step_idx, x)

    return x