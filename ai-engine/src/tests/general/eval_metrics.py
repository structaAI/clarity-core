import torch
import time
import numpy as np
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError
from diffusers import AutoencoderKL # type: ignore

# --- Hardware Setup ---
DEVICE = torch.device("cuda")
torch.backends.cudnn.benchmark = True 

def run_evaluation_session():
  print(f"\033[1mLocal Metric Evaluation Session\033[0m")
  print(f"Device: {torch.cuda.get_device_name(0)}")
  print("-" * 50)

  psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
  ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
  mse_metric = MeanSquaredError().to(DEVICE)

  vae = AutoencoderKL(
    in_channels=3, out_channels=3,
    block_out_channels=(64, 128), 
    latent_channels=4,
    down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
    up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
  ).to(DEVICE).eval() # type: ignore


  img = torch.randn(1, 3, 512, 512, device=DEVICE).clamp(-1, 1)

  start_event = torch.cuda.Event(enable_timing=True)
  end_event = torch.cuda.Event(enable_timing=True)

  start_event.record()
  with torch.no_grad():
    enc = vae.encode(img)
    post = getattr(enc, "latent_dist", enc[0])
    latents = post.sample()
    
    dec = vae.decode(latents) # type: ignore
    recon = getattr(dec, "sample", dec[0])
  end_event.record()

  torch.cuda.synchronize() # Wait for GPU to finish
  latency = start_event.elapsed_time(end_event)

  img_norm = (img + 1.0) / 2.0
  recon_norm = (recon + 1.0) / 2.0

  psnr_val = psnr_metric(recon_norm, img_norm)
  ssim_val = ssim_metric(recon_norm, img_norm)
  mse_val = mse_metric(recon_norm.flatten(), img_norm.flatten())

  mem_used = torch.cuda.max_memory_allocated() / (1024**2)

  # --- Results Table ---
  print(f"{'Metric':<25} | {'Value':<15}")
  print(f"{'-'*43}")
  print(f"{'Inference Latency':<25} | {latency:.2f} ms")
  print(f"{'PSNR (Fidelity)':<25} | {psnr_val.item():.2f} dB")
  print(f"{'SSIM (Structure)':<25} | {ssim_val.item():.4f}")
  print(f"{'MSE (Pixel Error)':<25} | {mse_val.item():.6f}")
  print(f"{'Peak VRAM Usage':<25} | {mem_used:.2f} MB")
  
  print(f"\n\033[1mLatent Space Sanity:\033[0m")
  print(f"Mean: {latents.mean().item():.4f} (Goal: ~0)")
  print(f"Std:  {latents.std().item():.4f} (Goal: ~1)")

if __name__ == "__main__":
  run_evaluation_session()