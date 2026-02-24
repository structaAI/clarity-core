import argparse
from pathlib import Path
import os
import sys
import logging
from dotenv import load_dotenv

import torch
from torch import Tensor
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL # type: ignore
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Any
from safetensors.torch import save_file

load_dotenv()

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

VAE_SCALE_FACTOR = os.getenv('VAE_SCALE_FACTOR') or 0.13025

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

class RawPairDataset(Dataset):
  def __init__(self, hr_dir: Path, lr_dir: Path, image_size: int) -> None:
    super().__init__()
    self.hr_dir, self.lr_dir = hr_dir, lr_dir
    
    hr_files = sorted(p for p in hr_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    if not hr_files:
      raise FileNotFoundError(f"No images in: {hr_dir}")

    self.pairs: list[tuple[Path, Path]] = []
    for hr_path in hr_files:
      lr_path = self._find_lr(lr_dir, hr_path.stem)
      if lr_path: self.pairs.append((hr_path, lr_path))

    log.info(f"Matched {len(self.pairs)} HR/LR pairs.")
    self.transform = transforms.Compose([
      transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5]),
    ])

  @staticmethod
  def _find_lr(lr_dir: Path, stem: str) -> Path | None:
    for ext in SUPPORTED_EXTENSIONS:
      candidate = lr_dir / f"{stem}{ext}"
      if candidate.exists(): return candidate
    return None

  def __len__(self) -> int: return len(self.pairs)

  def __getitem__(self, idx: int) -> tuple[Any, Any, str]:
    hr_path, lr_path = self.pairs[idx]
    hr = self.transform(Image.open(hr_path).convert("RGB"))
    lr = self.transform(Image.open(lr_path).convert("RGB"))
    return hr, lr, hr_path.stem
  
@torch.inference_mode()
def encode_batch(vae: AutoencoderKL, images: Tensor, device: torch.device, scale: float) -> Tensor:
    # Use BF16 for Blackwell architecture stability
    images = images.to(device, dtype=torch.bfloat16, non_blocking=True)

    enc_output = vae.encode(images)

    if isinstance(enc_output, tuple):
        posterior = enc_output[0]
    else:
        posterior = enc_output.latent_dist

    latents = posterior.sample() * scale
    return latents.to(torch.float32).cpu() # Return as float32 for storage precision

def cache_latents(args: argparse.Namespace) -> None:
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  
  # Auto-detect Scaling Factor
  # SDXL VAE uses 0.13025; standard SD-VAE uses 0.18215
  scale = VAE_SCALE_FACTOR
  log.info(f"Using VAE Scale Factor: {scale}")

  # Load VAE in BFloat16 for 5070 Ti optimization
  vae = AutoencoderKL.from_pretrained(
      args.vae_id, 
      torch_dtype=torch.bfloat16, 
      use_safetensors=True,
      local_files_only=True # Optimized for your local weights
  )
  vae.to(device) # type: ignore
  vae.eval()

  output_dir = Path(args.output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)

  dataset = RawPairDataset(Path(args.hr_dir), Path(args.lr_dir), args.image_size)
  loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

  for hr_batch, lr_batch, stems in tqdm(loader, desc="Encoding"):
    hr_latents = encode_batch(vae, hr_batch, device, scale) # type: ignore
    lr_latents = encode_batch(vae, lr_batch, device, scale) # type: ignore

    for i, stem in enumerate(stems):
      # Save using Safetensors for zero-copy training
      out_path = output_dir / f"latent_{stem}.safetensors"
      save_file({ 
        "hr": hr_latents[i], 
        "lr": lr_latents[i]
      }, str(out_path))

# ---------------------------------------------------------------------------
# CLI Setup
# ---------------------------------------------------------------------------
def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--hr_dir", required=True)
  parser.add_argument("--lr_dir", required=True)
  parser.add_argument("--output_dir", required=True)
  parser.add_argument("--vae_id", default="stabilityai/sdxl-vae") # Defaulting to your SDXL path
  parser.add_argument("--image_size", type=int, default=512)
  parser.add_argument("--batch_size", type=int, default=4)
  parser.add_argument("--num_workers", type=int, default=4)
  return parser.parse_args()

if __name__ == "__main__":
  cache_latents(parse_args())