import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import torch

from src.models.swin_dit.backbone import SwinDiT
from src.utils.config_manager_swin_dit import load_config

def run_sanity_check():
  try:
    config = load_config("D://Structa//claritycore//ai-engine//src//configs//swin_dit_config.yaml")
    print("Config Loaded Successfully")
  except Exception as e:
    raise RuntimeError(f"Could not Load Config File: {e}")
  
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model = SwinDiT(config).to(device=device)
  model.eval()
  print(f"Model instantiated on {device}")
 
 # Test-2 Executed till here

  batch_size = 2
  latent_size = config.model.latent_size # 64
  dummy_latents = torch.randn(batch_size, 4, latent_size, latent_size).to(device)
  dummy_timesteps = torch.randint(0, 1000, (batch_size,)).to(device)

  print("Running the Forward Pass")
  try:
    with torch.no_grad():
      output = model(dummy_latents, dummy_timesteps)
      print(f"\n--- Results ---")
      print(f"Input Shape:  {dummy_latents.shape}")
      print(f"Output Shape: {output.shape}")

      assert output.shape == dummy_latents.shape, "Output shape mismatch!"

      if torch.isnan(output).any():
        raise ValueError("NaN values obtained!")
      else:
        print("No NaN values obtained")

  except Exception as e:
    raise RuntimeError(f"Error during Forward Pass: {e}")


if __name__ == "__main__":
  run_sanity_check()