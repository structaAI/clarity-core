# Imports
import torch
from diffusers import AutoencoderKL # type: ignore

# General Check for CPU/GPU Availability
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Huggingface Model Card Information for VAE
MODEL_CARD_INFO = "stabilityai/sdxl-vae" 

try:
  # Loading the VAE model from Huggingface Hub
  vae = AutoencoderKL.from_pretrained(MODEL_CARD_INFO, non_blocking=False, device_map="auto").to(DEVICE) # pyright: ignore[reportArgumentType]
  vae.save_pretrained("saved_models")
  print(vae.device)
except Exception as e:
  # In case of failure, raise an error
  raise RuntimeError(f"Failed to Load VAE from Hugginface Hub: {e}")

