import os
import torch
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL # type: ignore

device = "cuda" if torch.cuda.is_available() else "cpu"

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
vae.eval()

transform = transforms.Compose([
  transforms.Resize((512, 512)),
  transforms.ToTensor(),
  transforms.Normalize([0.5], [0.5])
])

def convert_png_to_pt(hr_dir, lr_dir, output_dir):
  os.makedirs(output_dir, exist_ok=True)
  fnames = [f for f in os.listdir(hr_dir) if f.endswith('.png')]

  print(f"Starting conversion of {len(fnames)} image pairs...")
  
  for idx, fname in enumerate(fnames):
    # Load HR and LR pairs from your datasets folder
    hr_path = os.path.join(hr_dir, fname)
    lr_path = os.path.join(lr_dir, fname)

    hr_img = transform(Image.open(hr_path).convert('RGB')).unsqueeze(0).to(device)
    lr_img = transform(Image.open(lr_path).convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
      # Encode to 64x64x4 latent space
      # 0.18215 is the standard scaling factor for Stable Diffusion VAEs
      latents_hr = vae.encode(hr_img).latent_dist.sample() * 0.18215
      latents_lr = vae.encode(lr_img).latent_dist.sample() * 0.18215

    # 3. Save as a single dictionary for training
    # This saves storage space and makes loading during training much faster
    torch.save({
      'hr': latents_hr.cpu(),
      'lr': latents_lr.cpu()
    }, f"{output_dir}/latent_{idx}.pt")

    if idx % 100 == 0:
      print(f"Processed {idx} pairs...")

print(device)