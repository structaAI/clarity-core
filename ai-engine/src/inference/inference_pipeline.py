from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Callable, Optional, Union

import torch
import torchvision.transforms as T
from PIL import Image
from diffusers import AutoencoderKL
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from dotenv import load_dotenv

# 1. Path Management: Ensure we can see 'src'
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.diffusion.diffusion_engine import GaussianDiffusion
from src.models.swin_dit.backbone import SwinDiT
from src.utils.config_manager_swin_dit import load_config
from src.utils.pipeline_config import PipelineConfig

log = logging.getLogger(__name__)

# --- Helper Utilities ---

def _resolve_dtype(name: str) -> torch.dtype:
    _DTYPE_MAP = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    return _DTYPE_MAP.get(name, torch.float32)

def _image_to_tensor(image: Union[Image.Image, torch.Tensor], size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        t = image.to(device=device, dtype=dtype)
        if t.ndim == 3: t = t.unsqueeze(0)
        return t
    
    transform = T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.LANCZOS),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return transform(image.convert("RGB")).unsqueeze(0).to(device=device, dtype=dtype)

def _tensor_to_image(t: torch.Tensor) -> Image.Image:
    if t.ndim == 4: t = t.squeeze(0)
    # Clamp and convert from [-1, 1] back to [0, 255]
    t = ((t.cpu().float().detach().clamp(-1, 1) + 1.0) * 127.5).byte()
    return T.ToPILImage()(t)

# --- InferencePipeline ---

class InferencePipeline:
    def __init__(
        self,
        denoiser: SwinDiT,
        diffusion: GaussianDiffusion,
        config: PipelineConfig,
        vae: Optional[AutoencoderKL] = None,
        clip_model: Optional[PreTrainedModel] = None,
        clip_tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        self.denoiser = denoiser
        self.diffusion = diffusion
        self.config = config
        self.vae = vae
        self.clip_model = clip_model
        self.clip_tokenizer = clip_tokenizer

        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.dtype = _resolve_dtype(config.dtype)

        # Move Denoiser and CLIP to requested dtype (e.g., bfloat16)
        self.denoiser.to(self.device, self.dtype).eval()
        if self.clip_model is not None:
            self.clip_model.to(self.device, self.dtype).eval()

        # VAE is forced to Float32 for numerical stability and bias-type matching
        if self.vae is not None:
            self.vae.to(self.device, torch.float32).eval()
            print("⚙️ VAE initialized in Float32 mode for precision.")

    @classmethod
    def from_config(cls, config_path: str) -> "InferencePipeline":
        # Load environment variables from .env.local
        dotenv_path = project_root / ".env.local"
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path)
            print(f"✅ Loaded environment from: {dotenv_path}")

        cfg = PipelineConfig.from_yaml(config_path)
        swin_config = load_config(config_path)

        # 1. Initialize Components
        denoiser = SwinDiT(swin_config)
        diffusion = GaussianDiffusion(
            num_timesteps=cfg.num_timesteps,
            schedule=cfg.noise_schedule,
        )

        # 2. Load VAE
        vae = None
        path_to_vae = getattr(cfg, "vae_path", None)
        if path_to_vae:
            v_path = Path(path_to_vae).resolve()
            if v_path.exists():
                try:
                    vae = AutoencoderKL.from_pretrained(str(v_path), local_files_only=True)
                    print(f"✅ VAE Loaded successfully from {v_path.name}")
                except Exception:
                    # Fallback to single file load
                    weights = v_path / "diffusion_pytorch_model.safetensors"
                    if weights.exists():
                        vae = AutoencoderKL.from_single_file(str(weights), local_files_only=True)
                        print(f"✅ VAE Loaded via single-file fallback.")
            else:
                print(f"❌ VAE Path not found: {v_path}")

        # 3. Load CLIP (SigLIP)
        clip_model, clip_tokenizer = None, None
        path_to_clip = getattr(cfg, "clip_model_path", None)
        if path_to_clip:
            c_path = Path(path_to_clip).resolve()
            if c_path.exists():
                try:
                    clip_model = AutoModel.from_pretrained(str(c_path), local_files_only=True)
                    clip_tokenizer = AutoTokenizer.from_pretrained(str(c_path), local_files_only=True)
                    print(f"✅ CLIP Loaded.")
                except Exception as e:
                    print(f"⚠️ CLIP load failed: {e}")

        return cls(denoiser, diffusion, cfg, vae, clip_model, clip_tokenizer)

    def _require_vae(self) -> AutoencoderKL:
        if self.vae is None:
            raise RuntimeError("A VAE is required for pixel-space operations. Check VAE_PATH.")
        return self.vae

    @torch.no_grad()
    def _encode_prompt(self, prompt: str) -> Optional[torch.Tensor]:
        if not self.clip_model or not self.clip_tokenizer:
            return None
        inputs = self.clip_tokenizer([prompt], return_tensors="pt", padding=True).to(self.device)
        return self.clip_model.get_text_features(**inputs).to(dtype=self.dtype)

    @torch.no_grad()
    def encode_image(self, image: Union[Image.Image, torch.Tensor]) -> torch.Tensor:
        vae = self._require_vae()
        # Pixels must be Float32 for the VAE
        pixel = _image_to_tensor(image, self.config.image_size, self.device, torch.float32)
        
        out = vae.encode(pixel)
        posterior = out.latent_dist if hasattr(out, 'latent_dist') else out[0]
        
        # Sample and then cast back to bfloat16 for the SwinDiT
        latent = posterior.sample() * self.config.vae_scale_factor
        return latent.to(self.dtype)

    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> Image.Image:
        vae = self._require_vae()
        # Decode must be in Float32
        latent_f32 = latent.to(device=self.device, dtype=torch.float32)
        decoded = vae.decode(latent_f32 / self.config.vae_scale_factor).sample
        return _tensor_to_image(decoded)

    @torch.no_grad()
    def super_resolve(
        self, 
        lr_image: Union[Image.Image, torch.Tensor], 
        prompt: Optional[str] = None, 
        num_inference_steps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        # returns bfloat16
        x_lr = self.encode_image(lr_image) if not isinstance(lr_image, torch.Tensor) else lr_image.to(self.device, self.dtype)
        clip_embed = self._encode_prompt(prompt) if prompt else None
        
        # Model closure with slicing logic
        def model_fn(x, t):
            x_input = torch.cat([x, x_lr], dim=1)
            # Backbone now handles .to(dtype) internal casting
            out = self.denoiser(x, t, clip_embeddings=clip_embed)
            # If model predicts 8 channels, slice the first 4 for HR noise
            if out.shape[1] == 8:
                return out[:, :4, :, :]
            return out

        return self.diffusion.p_sample_loop(
            model_fn=model_fn,
            shape=(x_lr.shape[0], 4, x_lr.shape[2], x_lr.shape[3]), # We only want 4ch noise
            device=self.device,
            x_lr=x_lr,
            num_inference_steps=num_inference_steps or self.config.num_inference_steps,
            **kwargs
        )

    @torch.no_grad()
    def super_resolve_to_image(self, lr_image: Union[Image.Image, torch.Tensor], **kwargs) -> Image.Image:
        latent = self.super_resolve(lr_image, **kwargs)
        return self.decode_latent(latent)