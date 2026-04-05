"""
InferencePipeline — Unified orchestrator for conditional and unconditional
image denoising and super-resolution.

Public API
----------
    pipeline = InferencePipeline.from_config("configs/swin_dit_config.yaml")

    # Mode A: Unconditional denoising (pure noise → clean latent)
    latent = pipeline.denoise(batch_size=1)

    # Mode B: Unconditional SR (LR image → HR latent)
    latent = pipeline.super_resolve(lr_image)

    # Mode C: Text-conditioned SR
    latent = pipeline.super_resolve(lr_image, prompt="sharp building facade")

    # Mode D: Fully conditioned SR (text + degradation metadata)
    latent = pipeline.super_resolve(
        lr_image,
        prompt="sharp building facade",
        degradation_type=0,   # e.g. 0 = blur
        severity=0.7,
    )

    # Decode any latent to a PIL image
    image = pipeline.decode_latent(latent)

All four modes share the same model — conditioning is optional at every stage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
import torchvision.transforms as T
from PIL import Image
from diffusers import AutoencoderKL  # type: ignore
from transformers import AutoModel, AutoTokenizer  # type: ignore

from src.models.diffusion.diffusion_engine import GaussianDiffusion
from src.models.swin_dit.backbone import SwinDiT
from src.utils.config_manager_swin_dit import load_config
from src.utils.pipeline_config import PipelineConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: resolve torch dtype
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
  "float32": torch.float32,
  "bfloat16": torch.bfloat16,
  "float16": torch.float16,
}


def _resolve_dtype(name: str) -> torch.dtype:
  if name not in _DTYPE_MAP:
    raise ValueError(f"Unsupported dtype '{name}'. Choose from {list(_DTYPE_MAP)}")
  return _DTYPE_MAP[name]


# ---------------------------------------------------------------------------
# Image ↔ Tensor utilities
# ---------------------------------------------------------------------------

def _image_to_tensor(image: Union[Image.Image, torch.Tensor], size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
  """
  Convert a PIL image or a pre-built tensor to a normalised [0, 1] → [-1, 1]
  float tensor of shape [1, 3, size, size] on the target device.
  """
  if isinstance(image, torch.Tensor):
    t = image.to(device=device, dtype=dtype)
    if t.ndim == 3:
      t = t.unsqueeze(0)
    return t

  transform = T.Compose([
    T.Resize((size, size), interpolation=T.InterpolationMode.LANCZOS),
    T.ToTensor(),
    T.Normalize([0.5], [0.5]),
  ])
  return transform(image.convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype) #type: ignore


def _tensor_to_image(t: torch.Tensor) -> Image.Image:
  """
  Convert a [1, 3, H, W] or [3, H, W] tensor in [-1, 1] to a PIL image.
  """
  if t.ndim == 4:
    t = t.squeeze(0)
  t = ((t.cpu().float().clamp(-1, 1) + 1.0) * 127.5).byte()
  return T.ToPILImage()(t)


# ---------------------------------------------------------------------------
# InferencePipeline
# ---------------------------------------------------------------------------

class InferencePipeline:
  """
  Unified inference pipeline for SwinDiT + GaussianDiffusion.

  Construct via ``from_config`` (recommended) or directly in tests.

  Parameters
  ----------
  denoiser : SwinDiT
      The trained denoiser backbone.
  diffusion : GaussianDiffusion
      Scheduler with matching num_timesteps and noise_schedule.
  config : PipelineConfig
      Runtime configuration (paths, device, precision, …).
  vae : AutoencoderKL, optional
      VAE for pixel-space encode / decode.  When None, methods that require
      it raise RuntimeError with a clear message.
  clip_model : AutoModel, optional
      CLIP / SigLIP encoder for text conditioning.
  clip_tokenizer : AutoTokenizer, optional
      Tokenizer for the CLIP model.
  """

  def __init__(
    self,
    denoiser: SwinDiT,
    diffusion: GaussianDiffusion,
    config: PipelineConfig,
    vae: Optional[AutoencoderKL] = None,
    clip_model: Optional[AutoModel] = None,
    clip_tokenizer: Optional[AutoTokenizer] = None,
  ) -> None:
    self.denoiser = denoiser
    self.diffusion = diffusion
    self.config = config
    self.vae = vae
    self.clip_model = clip_model
    self.clip_tokenizer = clip_tokenizer

    self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    self.dtype = _resolve_dtype(config.dtype)

    self.denoiser.to(self.device, self.dtype).eval()
    if self.vae is not None:
      self.vae.to(self.device, self.dtype).eval()
    if self.clip_model is not None:
      self.clip_model.to(self.device, self.dtype).eval()

  # ------------------------------------------------------------------
  # Factory
  # ------------------------------------------------------------------

  @classmethod
  def from_config(cls, config_path: str) -> "InferencePipeline":
    """
    Build the full pipeline from a YAML config path.

    Paths for VAE, SwinDiT weights, and CLIP are read from environment
    variables (VAE_PATH, SWINDIT_WEIGHTS, CLIP_MODEL_SAVE_PATH).  Any
    component with an empty path is skipped and that capability is
    disabled at runtime.
    """
    cfg = PipelineConfig.from_yaml(config_path)
    cfg.validate()

    swin_config = load_config(config_path)

    # --- Denoiser ---
    denoiser = SwinDiT(swin_config)
    if cfg.swin_dit_weights:
      state = torch.load(cfg.swin_dit_weights, map_location="cpu", weights_only=True)
      denoiser.load_state_dict(state)
      log.info(f"Loaded SwinDiT weights from '{cfg.swin_dit_weights}'")
    else:
      log.warning("No SwinDiT weights path configured — using random initialisation.")

    # --- Diffusion scheduler ---
    diffusion = GaussianDiffusion(
      num_timesteps=cfg.num_timesteps,
      schedule=cfg.noise_schedule,
    )

    # --- VAE ---
    vae = None
    if cfg.vae_path:
      try:
        vae = AutoencoderKL.from_pretrained(
          cfg.vae_path, local_files_only=True
        )
        log.info(f"Loaded VAE from '{cfg.vae_path}'")
      except Exception as exc:
        log.warning(f"Could not load VAE: {exc}. Pixel-space methods disabled.")

    # --- CLIP ---
    clip_model, clip_tokenizer = None, None
    if cfg.clip_model_path:
      try:
        clip_model = AutoModel.from_pretrained(cfg.clip_model_path, local_files_only=True)
        clip_tokenizer = AutoTokenizer.from_pretrained(
          cfg.clip_model_path, local_files_only=True
        )
        log.info(f"Loaded CLIP from '{cfg.clip_model_path}'")
      except Exception as exc:
        log.warning(f"Could not load CLIP: {exc}. Text conditioning disabled.")

    return cls(
      denoiser=denoiser,
      diffusion=diffusion,
      config=cfg,
      vae=vae,
      clip_model=clip_model,
      clip_tokenizer=clip_tokenizer,
    )

  # ------------------------------------------------------------------
  # CLIP encoding
  # ------------------------------------------------------------------

  @torch.no_grad()
  def _encode_prompt(self, prompt: str) -> Optional[torch.Tensor]:
    """
    Encode a text prompt with the CLIP model.

    Returns None when CLIP is not configured, so callers can pass it
    directly to the denoiser as the clip_embeddings argument.
    """
    if self.clip_model is None or self.clip_tokenizer is None:
      log.debug("CLIP not loaded — running without text conditioning.")
      return None

    inputs = self.clip_tokenizer([prompt], return_tensors="pt", padding=True, truncation=True,).to(self.device)

    outputs = self.clip_model(**inputs)

    # SigLIP and standard CLIP both expose pooler_output or last_hidden_state
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
      embed = outputs.pooler_output
    else:
      embed = outputs.last_hidden_state[:, 0, :]

    return embed.to(dtype=self.dtype)  # [1, clip_dim]

  # ------------------------------------------------------------------
  # VAE encode / decode
  # ------------------------------------------------------------------

  def _require_vae(self) -> AutoencoderKL:
    if self.vae is None:
      raise RuntimeError(
        "A VAE is required for pixel-space operations. Set VAE_PATH in "
        "your environment or pass vae= to the constructor."
      )
    return self.vae

  @torch.no_grad()
  def encode_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
    """
    Encode a PIL image or RGB tensor to a VAE latent.

    Parameters
    ----------
    image : PIL.Image | torch.Tensor
        Input image.  Tensors should be [1, 3, H, W] in [-1, 1].

    Returns
    -------
    torch.Tensor
        Latent of shape [1, 4, latent_size, latent_size].
    """
    vae = self._require_vae()
    pixel = _image_to_tensor(image, self.config.image_size, self.device, self.dtype)
    posterior = vae.encode(pixel).latent_dist
    return posterior.sample() * self.config.vae_scale_factor

  @torch.no_grad()
  def decode_latent(self, latent: torch.Tensor) -> Image.Image:
    """
    Decode a VAE latent to a PIL image.

    Parameters
    ----------
    latent : torch.Tensor
        Shape [1, 4, H, W].

    Returns
    -------
    PIL.Image
    """
    vae = self._require_vae()
    latent = latent.to(device=self.device, dtype=self.dtype)
    decoded = vae.decode(latent / self.config.vae_scale_factor).sample
    return _tensor_to_image(decoded)

  # ------------------------------------------------------------------
  # Model closure builder
  # ------------------------------------------------------------------

  def _build_model_fn(
    self,
    clip_embeddings: Optional[torch.Tensor],
    degradation_type: Optional[int],
    severity: Optional[float],
    batch_size: int,
  ) -> Callable:
    """
    Build a (x_t, t) → ε closure that injects all conditioning tensors.

    All conditioning is captured via closure so ``p_sample_loop`` receives
    a clean two-argument function.
    """
    deg_type_tensor: Optional[torch.Tensor] = None
    if degradation_type is not None:
      deg_type_tensor = torch.tensor(
        [degradation_type] * batch_size,
        device=self.device,
        dtype=torch.long,
      )

    severity_tensor: Optional[torch.Tensor] = None
    if severity is not None:
      severity_tensor = torch.tensor(
        [severity] * batch_size,
        device=self.device,
        dtype=self.dtype,
      )

    def model_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
      return self.denoiser(
        x_t.to(dtype=self.dtype),
        t,
        clip_embeddings=clip_embeddings,
        degradation_type=deg_type_tensor,
        severity=severity_tensor,
      )

    return model_fn

  # ------------------------------------------------------------------
  # Public inference methods
  # ------------------------------------------------------------------

  @torch.no_grad()
  def denoise(
    self,
    batch_size: int = 1,
    prompt: Optional[str] = None,
    degradation_type: Optional[int] = None,
    severity: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
  ) -> torch.Tensor:
    """
    Unconditional (or conditionally guided) denoising from pure noise.

    Parameters
    ----------
    batch_size : int
        Number of latents to generate in parallel.
    prompt : str, optional
        Text prompt for CLIP-guided generation.
    degradation_type : int, optional
        Integer degradation class (0–4 with default config).
    severity : float, optional
        Severity in [0, 1].
    num_inference_steps : int, optional
        Override the config default.
    progress_callback : Callable, optional
        Called after each step with (step_index, current_x_tensor).

    Returns
    -------
    torch.Tensor
        Denoised latent, shape [B, 4, latent_size, latent_size].
    """
    clip_embed = self._encode_prompt(prompt) if prompt else None
    if clip_embed is not None and clip_embed.shape[0] == 1 and batch_size > 1:
      clip_embed = clip_embed.expand(batch_size, -1)

    model_fn = self._build_model_fn(
      clip_embeddings=clip_embed,
      degradation_type=degradation_type,
      severity=severity,
      batch_size=batch_size,
    )

    shape = (
      batch_size,
      self.config.in_channels,
      self.config.latent_size,
      self.config.latent_size,
    )

    return self.diffusion.p_sample_loop(
      model_fn=model_fn,
      shape=shape,
      device=self.device,
      x_lr=None,
      num_inference_steps=num_inference_steps or self.config.num_inference_steps,
      progress_callback=progress_callback,
    )

  @torch.no_grad()
  def super_resolve(
    self,
    lr_image: Union[Image.Image, torch.Tensor],
    prompt: Optional[str] = None,
    degradation_type: Optional[int] = None,
    severity: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
  ) -> torch.Tensor:
    """
    Super-resolve a low-resolution image.

    The LR image is encoded to a latent and used as a seeded starting
    point for the reverse diffusion loop, biasing generation toward the
    LR content while recovering high-frequency detail.

    Parameters
    ----------
    lr_image : PIL.Image | torch.Tensor
        Low-resolution input. Tensors must be [1, 3, H, W] in [-1, 1].
    prompt : str, optional
        Text prompt for CLIP-guided SR.
    degradation_type : int, optional
        Integer degradation class.
    severity : float, optional
        Severity in [0, 1].
    num_inference_steps : int, optional
        Override the config default.
    progress_callback : Callable, optional
        Called after each step.

    Returns
    -------
    torch.Tensor
        Super-resolved latent, shape [1, 4, latent_size, latent_size].
    """
    x_lr = self.encode_image(lr_image)          # [1, 4, H, W]
    batch_size = x_lr.shape[0]

    clip_embed = self._encode_prompt(prompt) if prompt else None

    model_fn = self._build_model_fn(
      clip_embeddings=clip_embed,
      degradation_type=degradation_type,
      severity=severity,
      batch_size=batch_size,
    )

    shape = (
      batch_size,
      self.config.in_channels,
      self.config.latent_size,
      self.config.latent_size,
    )

    return self.diffusion.p_sample_loop(
      model_fn=model_fn,
      shape=shape,
      device=self.device,
      x_lr=x_lr,
      num_inference_steps=num_inference_steps or self.config.num_inference_steps,
      progress_callback=progress_callback,
    )

  @torch.no_grad()
  def super_resolve_to_image(
    self,
    lr_image: Union[Image.Image, torch.Tensor],
    prompt: Optional[str] = None,
    degradation_type: Optional[int] = None,
    severity: Optional[float] = None,
    num_inference_steps: Optional[int] = None,
  ) -> Image.Image:
    """
    End-to-end convenience method: LR image → PIL HR image.

    Requires a loaded VAE.  For latent-only usage, call ``super_resolve``
    directly and decode with ``decode_latent`` yourself.
    """
    latent = self.super_resolve(
      lr_image,
      prompt=prompt,
      degradation_type=degradation_type,
      severity=severity,
      num_inference_steps=num_inference_steps,
    )
    return self.decode_latent(latent)