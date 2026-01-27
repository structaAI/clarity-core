import torch
import torchvision.transforms as T
from torchmetrics.image.psnr import PeakSignalNoiseRatio as psnr
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as ssim

import os
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL # type: ignore


BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIRECTORY, "saved_models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

vae=None

try:
  vae = AutoencoderKL.from_pretrained(
    MODEL_PATH, 
    local_files_only=True
  ).to(DEVICE).float()  # pyright: ignore[reportArgumentType]
  print(f"VAE loaded successfully to {DEVICE} in Full Precision (float32)")
except Exception as e:
  print(f"Error Loading VAE: {e}")

def verify_vae_setup():
  dummy_input = torch.randn(1, 3, 512, 512).to(DEVICE, dtype=torch.float32)

  with torch.no_grad():
    output = vae.encode(dummy_input) # type: ignore
    latents = output.latent_dist.sample() # pyright: ignore[reportAttributeAccessIssue]
    print(f"Successfully generated latents with shape: {latents.shape}")

def run_fidelity_test(img_tensor):
  with torch.no_grad():

    output = vae.encode(img_tensor.float()) # type: ignore
    latents = output.latent_dist.sample() # pyright: ignore[reportAttributeAccessIssue]
    
    dec_output = vae.decode(latents) # type: ignore
    reconstruction = dec_output.sample # pyright: ignore[reportAttributeAccessIssue]
    
    rec_np = reconstruction.detach().cpu().numpy()
    gt_np = img_tensor.detach().cpu().numpy()
    
    rec_np = ((rec_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    gt_np = ((gt_np + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    
    rec_final = torch.from_numpy(rec_np).float() / 255.0
    gt_final = torch.from_numpy(gt_np).float() / 255.0

    psnr_metric = psnr(data_range=1.0) 
    ssim_metric = ssim(data_range=1.0)
    
    psnr_val = psnr_metric(rec_final, gt_final).item()
    ssim_val = ssim_metric(rec_final, gt_final).item()

    return psnr_val, ssim_val

if __name__ == "__main__":
  test_img_path = "<File of Your Choice>.png"
  
  if not os.path.exists(test_img_path):
    print(f"Error: {test_img_path} not found in {os.getcwd()}")
  else:
    img = Image.open(test_img_path).convert("RGB")

    transform = T.Compose([
      T.Resize((512, 512)),
      T.ToTensor(),
      T.Normalize([0.5], [0.5]) 
    ])

    input_tensor = transform(img).unsqueeze(0).to(DEVICE, dtype=torch.float32) # pyright: ignore[reportAttributeAccessIssue]

    print(f"Running metrics on {test_img_path}...")
    psnr_value, ssim_value = run_fidelity_test(input_tensor)

    print("-" * 30)
    print(f"Results for {test_img_path}:")
    print(f"PSNR: {psnr_value:.2f} dB")
    print(f"SSIM: {ssim_value:.4f}")