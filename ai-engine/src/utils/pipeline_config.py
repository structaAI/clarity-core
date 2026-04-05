"""
PipelineConfig — Single source of truth for all inference-time configuration.

Replaces hardcoded path strings scattered across the codebase.  Can be built
from a YAML config file, from environment variables, or directly in code.

Usage
-----
    # From YAML (merges model hyperparams with runtime defaults)
    cfg = PipelineConfig.from_yaml("configs/swin_dit_config.yaml")

    # From environment (.env.local overrides)
    cfg = PipelineConfig.from_env()

    # Inline (unit tests / notebooks)
    cfg = PipelineConfig(vae_path="path/to/vae", swin_dit_weights="path/to/weights.pt")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import yaml


@dataclass
class PipelineConfig:
  # ------------------------------------------------------------------
  # Model paths
  # ------------------------------------------------------------------
  vae_path: str = ""
  """Local directory containing the AutoencoderKL safetensors checkpoint."""

  swin_dit_weights: str = ""
  """Path to a SwinDiT .pt checkpoint.  Empty string → random init."""

  clip_model_path: str = ""
  """Local directory containing the SigLIP / CLIP safetensors model.
  Empty string → CLIP conditioning disabled."""

  swin_dit_config: str = ""
  """Path to the YAML model config (swin_dit_config.yaml)."""

  # ------------------------------------------------------------------
  # Diffusion sampling
  # ------------------------------------------------------------------
  num_timesteps: int = 1000
  """Total diffusion steps the model was trained with."""

  num_inference_steps: int = 200
  """Steps to run at inference (< num_timesteps = faster / lower quality)."""

  noise_schedule: Literal["linear", "cosine"] = "linear"
  """Beta schedule type — must match training."""

  # ------------------------------------------------------------------
  # Latent / VAE
  # ------------------------------------------------------------------
  vae_scale_factor: float = 0.13025
  """SDXL-VAE latent scale factor.  Adjust if using a different VAE."""

  latent_size: int = 64
  """Spatial size of the latent feature map (H = W = latent_size)."""

  in_channels: int = 4
  """Number of VAE latent channels."""

  image_size: int = 512
  """Pixel-space target resolution."""

  # ------------------------------------------------------------------
  # Device
  # ------------------------------------------------------------------
  device: str = "cuda"
  """Torch device string.  Falls back to 'cpu' automatically if unavailable."""

  dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
  """Inference precision."""

  # ------------------------------------------------------------------
  # Constructors
  # ------------------------------------------------------------------

  @classmethod
  def from_yaml(cls, config_path: str) -> "PipelineConfig":
    """
    Load from the project YAML config, then override paths from env vars.

    The YAML file is expected to have the standard structure produced by
    ``config_manager_swin_dit.py``.  Runtime paths (vae_path, etc.) are
    read from environment variables so they are never committed to source.
    """
    with open(config_path, "r") as f:
      raw = yaml.safe_load(f)

    model_cfg = raw.get("model", {})
    diff_cfg = raw.get("diffusion", {})

    instance = cls(
      num_timesteps=diff_cfg.get("num_sampling_steps", 1000),
      noise_schedule=diff_cfg.get("noise_schedule", "linear"),
      latent_size=model_cfg.get("latent_size", 64),
      in_channels=model_cfg.get("in_channels", 4),
      swin_dit_config=config_path,
    )
    # Overlay environment variables for secrets / local paths
    instance._apply_env()
    return instance

  @classmethod
  def from_env(cls) -> "PipelineConfig":
    """Build config entirely from environment variables."""
    instance = cls()
    instance._apply_env()
    return instance

  def _apply_env(self) -> None:
    """Override fields from environment variables when present."""
    _str_fields = {
      "VAE_PATH": "vae_path",
      "SWINDIT_WEIGHTS": "swin_dit_weights",
      "CLIP_MODEL_SAVE_PATH": "clip_model_path",
      "SWINDIT_CONFIG": "swin_dit_config",
      "DEVICE": "device",
    }
    for env_key, attr in _str_fields.items():
      val = os.getenv(env_key)
      if val:
        setattr(self, attr, val)

    if os.getenv("VAE_SCALE_FACTOR"):
      self.vae_scale_factor = float(os.environ["VAE_SCALE_FACTOR"])
    if os.getenv("NUM_INFERENCE_STEPS"):
      self.num_inference_steps = int(os.environ["NUM_INFERENCE_STEPS"])

  # ------------------------------------------------------------------
  # Validation
  # ------------------------------------------------------------------

  def validate(self) -> None:
    """Raise informative errors for obviously wrong configurations."""
    if self.vae_path and not Path(self.vae_path).exists():
      raise FileNotFoundError(f"VAE path does not exist: '{self.vae_path}'")
    if self.swin_dit_weights and not Path(self.swin_dit_weights).exists():
      raise FileNotFoundError(f"SwinDiT weights not found: '{self.swin_dit_weights}'")
    if self.clip_model_path and not Path(self.clip_model_path).exists():
      raise FileNotFoundError(f"CLIP model path does not exist: '{self.clip_model_path}'")
    if self.num_inference_steps > self.num_timesteps:
        raise ValueError(f"num_inference_steps ({self.num_inference_steps}) cannot "
          f"exceed num_timesteps ({self.num_timesteps})."
        )

  def __repr__(self) -> str:  # pragma: no cover
    lines = ["PipelineConfig("]
    for k, v in self.__dict__.items():
      lines.append(f"  {k}={v!r},")
    lines.append(")")
    return "\n".join(lines)