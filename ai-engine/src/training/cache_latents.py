"""
cache_latents.py — True Super-Resolution Edition
=================================================
Encodes HR and LR images at DIFFERENT resolutions so the model must
genuinely learn spatial upscaling, not just image restoration.

Pipeline
--------
  HR image → resize to HR_SIZE (512) → VAE encode → [4, 64, 64]  saved as 'hr'
  LR image → resize to LR_SIZE (128) → VAE encode → [4, 16, 16]  saved as 'lr_small'

At training/inference time, lr_small is bilinear-upsampled to match the HR
latent spatial size before the channel concatenation:
  lr_up = F.interpolate(lr_small, size=(64,64), mode='bilinear')  → [4, 64, 64]
  x_in  = cat([x_t, lr_up], dim=1)                                → [8, 64, 64]

Why encode LR at 128px instead of upsampling pixels first?
  Upsampling pixels to 512px then encoding just gives the VAE a blurry image —
  the latent contains no information the HR latent doesn't already have.
  Encoding at the true 128px produces a latent [4,16,16] that captures the
  real low-frequency structure. The model must learn to hallucinate the
  missing high-frequency detail — that is super-resolution.

Scale factors
  HR_SIZE=512  → latent 512/8 = 64px spatial
  LR_SIZE=128  → latent 128/8 = 16px spatial
  Upsample ratio: 64/16 = 4×  →  true 4× SR in latent space

Environment variables (set in .env.local)
  VAE_PATH          — local AutoencoderKL directory
  HR_DIR            — directory of clean high-resolution images
  LR_DIR            — directory of degraded low-resolution images
  LATENT_CACHE_DIR  — output directory for .safetensors files
  HR_SIZE           — HR pixel size (default 512)
  LR_SIZE           — LR pixel size (default 128)
  VAE_SCALE_FACTOR  — default 0.13025 (SDXL-VAE)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from dotenv import load_dotenv
from diffusers import AutoencoderKL # type: ignore[import]
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
script_path  = Path(__file__).resolve()
project_root = script_path.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env.local")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SRPairDataset(Dataset):
    """
    Matches HR and LR images by filename stem and applies independent
    resize transforms so they are encoded at different resolutions.

    HR transform: resize to HR_SIZE (e.g. 512×512)
    LR transform: resize to LR_SIZE (e.g. 128×128)  ← true low resolution
    """

    def __init__(
        self,
        hr_dir:   Path,
        lr_dir:   Path,
        hr_size:  int,
        lr_size:  int,
    ) -> None:
        hr_map = {f.stem: f for f in hr_dir.iterdir()
                  if f.suffix.lower() in SUPPORTED_EXTENSIONS}
        lr_map = {f.stem: f for f in lr_dir.iterdir()
                  if f.suffix.lower() in SUPPORTED_EXTENSIONS}

        common = sorted(set(hr_map) & set(lr_map))
        self.pairs = [(hr_map[s], lr_map[s]) for s in common]

        log.info(f"Matched {len(self.pairs)} HR/LR pairs.")
        log.info(f"Orphans: {len(hr_map) - len(self.pairs)} HR, "
                 f"{len(lr_map) - len(self.pairs)} LR.")

        _norm = transforms.Normalize([0.5], [0.5])

        # HR: encode at full resolution
        self.hr_tf = transforms.Compose([
            transforms.Resize((hr_size, hr_size),
                              interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            _norm,
        ])

        # LR: encode at TRUE low resolution — this is the key change
        self.lr_tf = transforms.Compose([
            transforms.Resize((lr_size, lr_size),
                              interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            _norm,
        ])

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        hr_path, lr_path = self.pairs[idx]
        hr = self.hr_tf(Image.open(hr_path).convert("RGB"))
        lr = self.lr_tf(Image.open(lr_path).convert("RGB"))
        return hr, lr, hr_path.stem


# ---------------------------------------------------------------------------
# VAE encoding
# ---------------------------------------------------------------------------

@torch.inference_mode()
def encode_batch(
    vae:    AutoencoderKL,
    images: Tensor,
    device: torch.device,
    scale:  float,
) -> Tensor:
    """Encode a batch of pixel tensors to scaled VAE latents."""
    images = images.to(device, dtype=torch.float32, non_blocking=True)
    enc    = vae.encode(images)
    post   = enc.latent_dist if hasattr(enc, "latent_dist") else enc[0] # type: ignore
    return (post.sample() * scale).cpu().to(torch.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Cache SR latent pairs.")
    parser.add_argument("--vae_id",      default=os.getenv("VAE_PATH",          ""))
    parser.add_argument("--hr_dir",      default=os.getenv("HR_DIR",            ""))
    parser.add_argument("--lr_dir",      default=os.getenv("LR_DIR",            ""))
    parser.add_argument("--output_dir",  default=os.getenv("LATENT_CACHE_DIR",  ""))
    parser.add_argument("--hr_size",     default=int(os.getenv("HR_SIZE", "512")),   type=int)
    parser.add_argument("--lr_size",     default=int(os.getenv("LR_SIZE", "128")),   type=int)
    parser.add_argument("--scale",       default=float(os.getenv("VAE_SCALE_FACTOR", "0.13025")), type=float)
    parser.add_argument("--batch_size",  default=4, type=int)
    args = parser.parse_args()

    for name, val in [("--vae_id", args.vae_id), ("--hr_dir", args.hr_dir),
                      ("--lr_dir", args.lr_dir), ("--output_dir", args.output_dir)]:
        if not val:
            raise ValueError(f"{name} is not set. Add it to .env.local.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device     : {device}")
    log.info(f"HR size    : {args.hr_size}px  → latent {args.hr_size//8}×{args.hr_size//8}")
    log.info(f"LR size    : {args.lr_size}px  → latent {args.lr_size//8}×{args.lr_size//8}")
    log.info(f"SR factor  : {args.hr_size // args.lr_size}× in pixel space, "
             f"{(args.hr_size//8) // (args.lr_size//8)}× in latent space")
    log.info(f"Scale      : {args.scale}")

    # --- VAE ---
    vae = AutoencoderKL.from_pretrained(
        args.vae_id,
        torch_dtype=torch.float32,
        local_files_only=True,
    ).to(device).eval() # type: ignore

    # --- Dataset ---
    dataset = SRPairDataset(
        hr_dir=Path(args.hr_dir),
        lr_dir=Path(args.lr_dir),
        hr_size=args.hr_size,
        lr_size=args.lr_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=0,       # safetensors + Windows → num_workers=0
        pin_memory=True,
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    log.info(f"Encoding {len(dataset)} pairs → {args.output_dir}")
    for hr_batch, lr_batch, stems in tqdm(loader, desc="Encoding"):
        hr_latents = encode_batch(vae, hr_batch, device, args.scale)
        lr_latents = encode_batch(vae, lr_batch, device, args.scale)

        # hr_latents: [B, 4, hr_size//8, hr_size//8]  e.g. [B, 4, 64, 64]
        # lr_latents: [B, 4, lr_size//8, lr_size//8]  e.g. [B, 4, 16, 16]
        for i, stem in enumerate(stems):
            out = Path(args.output_dir) / f"latent_{stem}.safetensors"
            save_file(
                {
                    "hr":       hr_latents[i],   # [4, 64, 64]
                    "lr_small": lr_latents[i],   # [4, 16, 16]  ← true low-res
                },
                str(out),
            )

    log.info("Done. Re-run training with the new latent cache.")
    log.info(f"HR latent shape: [4, {args.hr_size//8}, {args.hr_size//8}]")
    log.info(f"LR latent shape: [4, {args.lr_size//8}, {args.lr_size//8}]  "
             f"(upsampled to HR size at train/inference time)")


if __name__ == "__main__":
    main()