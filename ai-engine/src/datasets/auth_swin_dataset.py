"""
auth_swin_dataset.py — True SR Edition
=======================================
Loads latent pairs encoded by the new cache_latents.py.

Key change: the dataset now stores 'lr_small' at the true low resolution
(e.g. [4, 16, 16]) instead of 'lr' at the full HR resolution ([4, 64, 64]).

At __getitem__ time, lr_small is bilinear-upsampled to match the HR latent
spatial size. This is the correct approach because:

  - The model receives a 4× upsampled latent as its LR conditioning input.
  - The gap between the upsampled LR and the clean HR is what the model
    learns to fill — that gap is genuine high-frequency detail, i.e. SR.
  - Bilinear upsampling in latent space is the standard approach used in
    latent diffusion SR (LDM, StableSR, etc.).

Backward compatibility
  If a .safetensors file has the old 'lr' key (same size as 'hr'), the
  dataset falls back to using it directly so existing caches still work.
  A warning is printed once per run so you know which mode is active.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset
from safetensors.torch import load_file


class AuthSwinDataset(Dataset):
    """
    Loads pre-encoded HR/LR latent pairs from a directory of .safetensors files.

    Expected keys in each file (new format):
      'hr'       — clean HR latent  [4, H,   W  ]  e.g. [4, 64, 64]
      'lr_small' — degraded LR latent [4, H/4, W/4]  e.g. [4, 16, 16]

    Deprecated key (old format, backward compatible):
      'lr'       — degraded LR latent at the same size as HR [4, H, W]

    Returns dict with:
      'hr' — [4, H, W]  clean HR latent  (target)
      'lr' — [4, H, W]  LR latent upsampled to HR spatial size (conditioning)
    """

    _warned_old_format: bool = False

    def __init__(self, latent_dir: str) -> None:
        self.latent_dir = Path(latent_dir)
        self.files = sorted(
            f for f in self.latent_dir.iterdir() if f.suffix == ".safetensors"
        )
        if not self.files:
            raise FileNotFoundError(
                f"No .safetensors files found in {latent_dir}. "
                "Run cache_latents.py first."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = load_file(str(self.files[idx]), device="cpu")

        hr: Tensor = data["hr"]   # [4, H, W]

        if "lr_small" in data:
            # New format: upsample low-res latent to HR spatial size
            lr_small: Tensor = data["lr_small"]  # [4, H/4, W/4]
            H, W = hr.shape[-2], hr.shape[-1]
            lr: Tensor = F.interpolate(
                lr_small.unsqueeze(0).float(),  # [1, 4, H/4, W/4]
                size=(H, W),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).to(hr.dtype)           # [4, H, W]

        elif "lr" in data:
            # Backward-compatible: old cache where LR was encoded at same size
            if not AuthSwinDataset._warned_old_format:
                warnings.warn(
                    "Dataset contains old-format 'lr' key (same spatial size as 'hr'). "
                    "The model will perform image restoration, NOT super-resolution. "
                    "Re-run cache_latents.py with LR_SIZE=128 to enable true 4× SR.",
                    UserWarning,
                    stacklevel=2,
                )
                AuthSwinDataset._warned_old_format = True
            lr = data["lr"]

        else:
            raise KeyError(
                f"File {self.files[idx].name} has neither 'lr_small' nor 'lr' key. "
                f"Keys found: {list(data.keys())}"
            )

        return {"hr": hr, "lr": lr}