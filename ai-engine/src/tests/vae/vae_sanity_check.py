import torch
from diffusers import AutoencoderKL # type: ignore

vae_local_path = "ai-engine//src//models//vae//saved_models"

device = "cuda" if torch.cuda.is_available() else "cpu"

vae = AutoencoderKL.from_pretrained(
  vae_local_path,
  local_files_only=True,
  use_safetensors=True,
  torch_dtype = torch.float16
).to(device) # type: ignore

vae.eval()
