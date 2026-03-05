import argparse
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

import torch
from diffusers import AutoencoderKL # type: ignore
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from safetensors.torch import save_file
from PIL import Image
from torchvision import transforms

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

class RawPairDataset(Dataset):
  def __init__(self, hr_dir: Path, lr_dir: Path, image_size: int):
    super().__init__()
    self.hr_dir, self.lr_dir = hr_dir, lr_dir
    hr_files = sorted(p for p in hr_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS)
    
    self.pairs = []
    for hr_path in hr_files:
      lr_path = self._find_lr(lr_dir, hr_path.stem)
      if lr_path: self.pairs.append((hr_path, lr_path))

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

  def __len__(self): return len(self.pairs)

  def __getitem__(self, idx):
    hr_p, lr_p = self.pairs[idx]
    hr = self.transform(Image.open(hr_p).convert("RGB"))
    lr = self.transform(Image.open(lr_p).convert("RGB"))
    return hr, lr, hr_p.stem

@torch.inference_mode()
def encode_batch(vae, images, device, scale):
  images = images.to(device, dtype=torch.bfloat16)
  enc_output = vae.encode(images)
  posterior = enc_output[0] if isinstance(enc_output, tuple) else enc_output.latent_dist
  return (posterior.sample() * scale).to(torch.float32).cpu()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--vae_id", default=os.getenv("VAE_PATH"))
  parser.add_argument("--hr_dir", default=os.getenv("HR_DIR"))
  parser.add_argument("--lr_dir", default=os.getenv("LR_DIR"))
  parser.add_argument("--output_dir", default=os.getenv("LATENT_CACHE_DIR"))
  args = parser.parse_args()

  device = torch.device("cuda")
  scale = float(os.getenv("VAE_SCALE_FACTOR", 0.13025))
  
  vae = AutoencoderKL.from_pretrained(args.vae_id, torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval() # type: ignore
  
  dataset = RawPairDataset(Path(args.hr_dir), Path(args.lr_dir), 512)
  loader = DataLoader(dataset, batch_size=4, num_workers=0, pin_memory=True)

  os.makedirs(args.output_dir, exist_ok=True)
  for hr_img, lr_img, stems in tqdm(loader, desc="Encoding"):
    hr_latents = encode_batch(vae, hr_img, device, scale)
    lr_latents = encode_batch(vae, lr_img, device, scale)
    for i, stem in enumerate(stems):
      save_file({"hr": hr_latents[i], "lr": lr_latents[i]}, os.path.join(args.output_dir, f"latent_{stem}.safetensors"))

if __name__ == "__main__":
  main()