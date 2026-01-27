import torch
from diffusers import AutoencoderKL # type: ignore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CARD_INFO = "stabilityai/sdxl-vae"

try:
  vae = AutoencoderKL.from_pretrained(MODEL_CARD_INFO, non_blocking=False, device_map="auto").to(DEVICE) # pyright: ignore[reportArgumentType]
  vae.save_pretrained("saved_models")
  print(vae.device)
except Exception as e:
  raise RuntimeError(f"Failed to Load VAE from Hugginface Hub: {e}")

