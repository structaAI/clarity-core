import os
import argparse
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

import torch
from torch import Tensor
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL # type: ignore
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from safetensors.torch import save_file

# --- Security & Environment ---
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ---------------------------------------------------------------------------
# Dataset with Automatic Pair Matching
# ---------------------------------------------------------------------------

class COCOPairDataset(Dataset):
  """
  Automatically matches HR and LR images by filename stem.
  This solves the 'random unzip order' issue by indexing both folders.
  """
  def __init__(self, hr_dir: Path, lr_dir: Path, image_size: int) -> None:
    super().__init__()
    self.hr_dir, self.lr_dir = hr_dir, lr_dir
    
    # 1. Index filenames in both directories
    hr_map = {f.stem: f for f in hr_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS}
    lr_map = {f.stem: f for f in lr_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS}

    # 2. Find common IDs (Set Intersection)
    common_stems = sorted(list(set(hr_map.keys()) & set(lr_map.keys())))
    
    self.pairs = [(hr_map[s], lr_map[s]) for s in common_stems]
    
    log.info(f"Successfully matched {len(self.pairs)} HR/LR pairs.")
    log.info(f"Orphans found: {len(hr_map) - len(self.pairs)} HR, {len(lr_map) - len(self.pairs)} LR.")

    self.transform = transforms.Compose([
      transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
      transforms.ToTensor(),
      transforms.Normalize([0.5], [0.5]),
    ])

  def __len__(self) -> int:
    return len(self.pairs)

  def __getitem__(self, idx: int):
    hr_path, lr_path = self.pairs[idx]
    hr = self.transform(Image.open(hr_path).convert("RGB"))
    lr = self.transform(Image.open(lr_path).convert("RGB"))
    return hr, lr, hr_path.stem

# ---------------------------------------------------------------------------
# VAE Encoding with BF16 Stability
# ---------------------------------------------------------------------------

@torch.inference_mode()
def encode_batch(vae: AutoencoderKL, images: Tensor, device: torch.device, scale: float) -> Tensor:
  # Use BF16 for Blackwell architecture stability (5070 Ti)
  images = images.to(device, dtype=torch.bfloat16, non_blocking=True)
  enc_output = vae.encode(images)

  # Defensive check for Tuple vs Object returns
  posterior = enc_output[0] if isinstance(enc_output, tuple) else enc_output.latent_dist
  
  latents = posterior.sample() * scale
  return latents.to(torch.float32).cpu()

# ---------------------------------------------------------------------------
# Execution Logic
# ---------------------------------------------------------------------------

def main():
  parser = argparse.ArgumentParser(description="Cache COCO latents securely.")
  parser.add_argument("--vae_id", default=os.getenv("VAE_PATH", "stabilityai/sdxl-vae"))
  parser.add_argument("--hr_dir", default=os.getenv("HR_DIR"))
  parser.add_argument("--lr_dir", default=os.getenv("LR_DIR"))
  parser.add_argument("--output_dir", default=os.getenv("LATENT_CACHE_DIR"))
  args = parser.parse_args()

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # Dynamic Scale Selection
  scale = float(os.getenv("VAE_SCALE_FACTOR", 0.13025))
  if "sdxl" not in args.vae_id.lower() and scale == 0.13025:
    log.warning("Non-SDXL VAE detected, but using 0.13025 scale. Double-check your .env!")

  # Load VAE locally
  vae = AutoencoderKL.from_pretrained(
    args.vae_id, 
    torch_dtype=torch.bfloat16, 
    local_files_only=True
  ).to(device).eval() # type: ignore

  dataset = COCOPairDataset(Path(args.hr_dir), Path(args.lr_dir), 512)
  
  # num_workers=0 is safer on Windows; use 2-4 if you're on Linux
  loader = DataLoader(dataset, batch_size=4, num_workers=0, pin_memory=True)

  os.makedirs(args.output_dir, exist_ok=True)

  for hr_batch, lr_batch, stems in tqdm(loader, desc="Encoding Latents"):
    hr_latents = encode_batch(vae, hr_batch, device, scale)
    lr_latents = encode_batch(vae, lr_batch, device, scale)

    for i, stem in enumerate(stems):
      out_path = Path(args.output_dir) / f"latent_{stem}.safetensors"
      save_file({"hr": hr_latents[i], "lr": lr_latents[i]}, str(out_path))

if __name__ == "__main__":
  main()