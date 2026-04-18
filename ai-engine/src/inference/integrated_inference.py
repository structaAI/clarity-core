"""
IntegratedInferencePipeline — Fixed.
=====================================
Supports both inference modes from a single integrated checkpoint:

  Mode A — Unconditional SR
      pipeline.super_resolve(lr_image)

  Mode B — Text-guided SR with CFG
      pipeline.super_resolve(lr_image, prompt="...", guidance_scale=7.5)

Bugs fixed vs previous version
-------------------------------
1. from_checkpoint used ckpt.get("model_state_dict", ckpt) as a fallback that
   would silently treat the entire checkpoint dict as a state dict when the key
   was absent. Fixed: explicit KeyError with a clear message.

2. _TextEncoder.encode() was decorated @torch.no_grad(), which blocked gradient
   flow through the projection during any training-context usage. Fixed: the
   decorator is removed; no_grad is applied only to the frozen backbone call
   internally. The method is safe for both inference (no grad needed) and any
   custom training loops (projection grads preserved).

3. Resolution/scaling documentation corrected: actual output resolution is
   determined by config.image_size, not hardcoded to 1024. At image_size=512,
   the VAE produces a [B,4,64,64] latent which decodes to 512×512 pixels.
   To get 1024×1024, set image_size=1024 and re-run cache_latents.py.

4. from_checkpoint read ckpt["projection_state_dict"] correctly only when
   "model_state_dict" was present in ckpt. Now uses explicit key presence checks
   for both keys independently, making loading robust to any checkpoint format.

5. TextEncoder now extracts text_model sub-module from the full SiglipModel,
   mirroring auth_train_engine.py exactly. This fixes AttributeError caused by
   get_text_features() returning BaseModelOutputWithPooling instead of a tensor,
   and ensures projection weights load correctly since they were trained against
   text_model output.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable, List, Optional, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from diffusers import AutoencoderKL # type: ignore[import]
from dotenv import load_dotenv
from transformers import AutoModel, AutoTokenizer

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env.local")

from src.models.diffusion.diffusion_engine import GaussianDiffusion
from src.models.swin_dit.backbone import SwinDiT
from src.utils.config_manager_swin_dit import load_config
from src.utils.pipeline_config import PipelineConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image ↔ Tensor helpers
# ---------------------------------------------------------------------------

def _to_tensor(
    image: Union[Image.Image, torch.Tensor],
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        t = image.to(device=device, dtype=dtype)
        return t.unsqueeze(0) if t.ndim == 3 else t
    tf = T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return tf(image.convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype)


def _to_pil(t: torch.Tensor) -> Image.Image:
    t = t.squeeze(0) if t.ndim == 4 else t
    t = ((t.cpu().float().detach().clamp(-1, 1) + 1.0) * 127.5).byte()
    return T.ToPILImage()(t)


# ---------------------------------------------------------------------------
# Text encoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """
    Frozen SigLIP / CLIP backbone with a trainable projection head.
    Mirrors auth_train_engine.py exactly so checkpoint weights load correctly.
    """

    def __init__(self, clip_path: str, model_dim: int) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(clip_path, local_files_only=True)
        full_model = AutoModel.from_pretrained(clip_path, local_files_only=True)

        # Mirror training TextEncoder exactly: isolate text_model to avoid
        # pixel_values requirements and ensure projection weights load correctly.
        if hasattr(full_model, "text_model"):
            self.backbone = full_model.text_model
        else:
            self.backbone = full_model

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            dummy = self.tokenizer(["probe"], return_tensors="pt", padding=True, truncation=True)
            output = self.backbone(**dummy)
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                probe = output.pooler_output
            elif hasattr(output, "last_hidden_state"):
                probe = output.last_hidden_state.mean(1)
            else:
                probe = output[0] if isinstance(output, (list, tuple)) else output
        clip_dim = probe.shape[-1]

        self.projection = nn.Sequential(
            nn.Linear(clip_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def encode(
        self,
        prompts: List[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        ).to(device)

        with torch.no_grad():
            output = self.backbone(**inputs)
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                raw = output.pooler_output
            else:
                raw = output.last_hidden_state.mean(1)

        return self.projection(raw.to(dtype))  # [B, model_dim]


# ---------------------------------------------------------------------------
# IntegratedInferencePipeline
# ---------------------------------------------------------------------------

class IntegratedInferencePipeline:
    """
    Unified SR pipeline. Loads a checkpoint saved by integrated_train.py.

    Resolution note
    ---------------
    Actual output resolution = config.image_size (not hardcoded 1024).
    The VAE downscales by 8, so:
      image_size=512  → latent [B,4,64,64]   → decoded 512×512
      image_size=1024 → latent [B,4,128,128] → decoded 1024×1024
    To get 1024×1024 output you must set image_size=1024 in the config AND
    re-encode your dataset with cache_latents.py at that resolution.
    """

    def __init__(
        self,
        denoiser:     SwinDiT,
        diffusion:    GaussianDiffusion,
        config:       PipelineConfig,
        vae:          Optional[AutoencoderKL] = None,
        text_encoder: Optional[TextEncoder]  = None,
    ) -> None:
        self.denoiser     = denoiser
        self.diffusion    = diffusion
        self.config       = config
        self.vae          = vae
        self.text_encoder = text_encoder

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype  = {"float32": torch.float32,
                       "bfloat16": torch.bfloat16,
                       "float16":  torch.float16}.get(config.dtype, torch.float32)

        self.denoiser.to(self.device, self.dtype).eval()

        if self.vae is not None:
            self.vae.to(self.device, torch.float32).eval()

        if self.text_encoder is not None:
            self.text_encoder.backbone   = self.text_encoder.backbone.to(self.device)
            self.text_encoder.projection = self.text_encoder.projection.to(self.device, self.dtype)
            self.text_encoder.projection.eval()

        for attr in [
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "sqrt_recip_alphas", "posterior_variance",
            "posterior_log_variance_clipped",
            "posterior_mean_coef1", "posterior_mean_coef2",
        ]:
            setattr(self.diffusion, attr,
                    getattr(self.diffusion, attr).to(self.device))

    @classmethod
    def from_checkpoint(
        cls,
        config_path: str,
        checkpoint:  str,
        clip_path:   Optional[str] = None,
        vae_path:    Optional[str] = None,
        device:      str = "cuda",
        dtype:       str = "bfloat16",
    ) -> "IntegratedInferencePipeline":
        swin_config = load_config(config_path)
        cfg         = PipelineConfig.from_yaml(config_path)
        cfg.device  = device
        cfg.dtype   = dtype  # type: ignore[assignment]
        if vae_path:
            cfg.vae_path = vae_path

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)

        denoiser = SwinDiT(swin_config)
        if "model_state_dict" not in ckpt:
            raise KeyError(
                f"Checkpoint '{checkpoint}' has no 'model_state_dict' key. "
                f"Keys found: {list(ckpt.keys())}"
            )
        denoiser.load_state_dict(ckpt["model_state_dict"], strict=True)
        log.info(f"Loaded denoiser from {Path(checkpoint).name}")

        diffusion = GaussianDiffusion(
            num_timesteps=swin_config.diffusion.num_sampling_steps,
            schedule=swin_config.diffusion.noise_schedule,
        )

        vae: Optional[AutoencoderKL] = None
        _vae_path = vae_path or os.getenv("VAE_PATH", "")
        if _vae_path and Path(_vae_path).exists():
            try:
                vae = AutoencoderKL.from_pretrained(_vae_path, local_files_only=True)
                log.info("VAE loaded.")
            except Exception as e:
                log.warning(f"VAE load failed: {e}")

        text_encoder: Optional[TextEncoder] = None
        _clip_path = clip_path or os.getenv("CLIP_MODEL_SAVE_PATH", "")
        if _clip_path and Path(_clip_path).exists():
            te = TextEncoder(clip_path=_clip_path, model_dim=swin_config.model.embed_dim)
            if "projection_state_dict" in ckpt:
                te.projection.load_state_dict(ckpt["projection_state_dict"])
                log.info("Loaded trained projection weights.")
            else:
                log.warning(
                    "No 'projection_state_dict' in checkpoint. "
                    "Text conditioning will use a random projection — quality will be poor."
                )
            text_encoder = te

        return cls(denoiser, diffusion, cfg, vae, text_encoder)

    def _require_vae(self) -> AutoencoderKL:
        if self.vae is None:
            raise RuntimeError(
                "VAE is required for pixel-space encode/decode. "
                "Pass vae_path= to from_checkpoint()."
            )
        return self.vae

    @torch.no_grad()
    def encode_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        vae   = self._require_vae()
        pixel = _to_tensor(image, self.config.image_size, self.device, torch.float32)
        out   = vae.encode(pixel)
        post  = out.latent_dist if hasattr(out, "latent_dist") else out[0]
        return (post.sample() * self.config.vae_scale_factor).to(self.dtype)

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        vae     = self._require_vae()
        l_f32   = latent.to(device=self.device, dtype=torch.float32)
        decoded = vae.decode(l_f32 / self.config.vae_scale_factor).sample
        return _to_pil(decoded)

    @torch.no_grad()
    def super_resolve(
        self,
        lr_image:            Union[Image.Image, torch.Tensor],
        prompt:              Optional[str] = None,
        guidance_scale:      float = 7.5,
        num_inference_steps: Optional[int] = None,
        lr_is_latent:        bool = False,
        progress_callback:   Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> torch.Tensor:
        x_lr = (
            lr_image.to(self.device, self.dtype)  # type: ignore[union-attr]
            if lr_is_latent
            else self.encode_image(lr_image)
        )
        B, C, H, W = x_lr.shape

        use_cfg = (
            prompt is not None
            and self.text_encoder is not None
            and guidance_scale != 1.0
        )
        cond_embed: Optional[torch.Tensor] = None
        if prompt is not None and self.text_encoder is not None:
            cond_embed = self.text_encoder.encode(
                [prompt] * B, device=self.device, dtype=self.dtype
            )

        def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            x_in = torch.cat([x, x_lr], dim=1)  # [B, 8, H, W]
            if use_cfg:
                uncond = self.denoiser(x_in, t, clip_embeddings=None)[:, :4]
                cond   = self.denoiser(x_in, t, clip_embeddings=cond_embed)[:, :4]
                return uncond + guidance_scale * (cond - uncond)
            else:
                out = self.denoiser(x_in, t, clip_embeddings=cond_embed)
                return out[:, :4] if out.shape[1] == 8 else out

        return self.diffusion.p_sample_loop(
            model_fn=model_fn,
            shape=(B, C, H, W),
            device=self.device,
            x_lr=x_lr,
            num_inference_steps=num_inference_steps or self.config.num_inference_steps,
            progress_callback=progress_callback,
        )

    @torch.no_grad()
    def super_resolve_to_image(
        self,
        lr_image: Union[Image.Image, torch.Tensor],
        **kwargs,
    ) -> Image.Image:
        return self.decode_latent(self.super_resolve(lr_image, **kwargs))


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import tempfile
    import yaml

    print("Running smoke test (CPU, no VAE, no CLIP)...")

    cfg_dict = {
        "model": {
            "name": "smoke-test", "latent_size": 8, "in_channels": 8,
            "patch_size": 2, "embed_dim": 32, "depths": [1, 1],
            "num_heads": [2, 2], "window_size": 4,
            "use_pswa_bridge": False, "bridge_alpha": 0.5,
        },
        "diffusion": {
            "num_sampling_steps": 10, "noise_schedule": "linear",
            "prediction_type": "epsilon",
        },
        "training": {"learning_rate": 1e-4, "precision": "fp32", "batch_size": 1},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(cfg_dict, tf)
        tmp_cfg = tf.name

    swin_cfg  = load_config(tmp_cfg)
    denoiser  = SwinDiT(swin_cfg)
    diffusion = GaussianDiffusion(num_timesteps=10, schedule="linear")
    cfg       = PipelineConfig(
        num_timesteps=10, num_inference_steps=5,
        latent_size=8, in_channels=4, image_size=64,
        device="cpu", dtype="float32",
    )

    pipeline = IntegratedInferencePipeline(denoiser, diffusion, cfg)

    x_lr = torch.randn(1, 4, 8, 8)
    out  = pipeline.super_resolve(x_lr, lr_is_latent=True)
    assert out.shape == (1, 4, 8, 8), f"Bad shape: {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    print(f"  Unconditional SR: {tuple(x_lr.shape)} → {tuple(out.shape)}  ✓")

    print("Smoke test passed.")
    os.unlink(tmp_cfg)
