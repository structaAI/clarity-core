import sys, time, argparse, torch
from pathlib import Path
from types import SimpleNamespace

# 1. Bootstrap: Locate ai-engine/src/
SRC = next((p for p in [Path(__file__).resolve(), *Path(__file__).resolve().parents] 
            if (p / "ai-engine/src").is_dir() or (p / "src/models").is_dir()), Path(__file__).parent)
SRC = SRC / "ai-engine/src" if (SRC / "ai-engine/src").is_dir() else SRC
if str(SRC) not in sys.path: sys.path.insert(0, str(SRC))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def log(tag, msg, success=True):
    symbol = "\033[92m✔\033[0m" if success else "\033[91m✘\033[0m"
    print(f"[{tag:10}] {symbol} {msg}")

# --- Component Demos ---

def demo_vae():
  from diffusers import AutoencoderKL # type: ignore
  try:
    vae = AutoencoderKL(
      in_channels=3, 
      out_channels=3, 
      down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
      up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
      block_out_channels=(32, 64), # First block expects 32, second expects 64
      latent_channels=4,
    ).to(DEVICE).eval() # type: ignore

    # Your test image must have 3 channels (RGB) to match in_channels=3
    img = torch.randn(1, 3, 512, 512, device=DEVICE) 
    
    with torch.no_grad():
      # Using the robust tuple/object check we discussed
      enc_out = vae.encode(img)
      posterior = getattr(enc_out, "latent_dist", enc_out[0])
      latents = posterior.sample()
      
      dec_out = vae.decode(latents) # type: ignore
      recon = getattr(dec_out, "sample", dec_out[0])
        
    log("VAE", f"Success. Latent: {tuple(latents.shape)}")
    return True
  except Exception as e: 
    return log("VAE", f"Architecture mismatch: {e}", False)

def demo_diffusion():
  from models.diffusion.diffusion_engine import GaussianDiffusion
  try:
    diff = GaussianDiffusion(num_timesteps=1000, schedule="linear")
    x0 = torch.randn(1, 4, 64, 64)
    t = torch.tensor([500])
    xt = diff.q_sample(x0, t)
    log("DIFFUSION", f"q_sample at t=500 preserved shape {tuple(xt.shape)}")
    return True
  except Exception as e: return log("DIFFUSION", e, False)

def demo_swindit():
  from models.swin_dit.backbone import SwinDiT
  try:
    cfg = SimpleNamespace(model=SimpleNamespace(latent_size=64, in_channels=4, patch_size=2, embed_dim=768, 
                          depths=[2,2,2,2], num_heads=[3,6,12,24], window_size=8, use_pswa_bridge=True))
    model = SwinDiT(cfg).to(DEVICE).eval()
    x, t = torch.randn(1, 4, 64, 64, device=DEVICE), torch.randint(0, 1000, (1,), device=DEVICE)
    with torch.no_grad(): out = model(x, t)
    log("SWINDIT", f"Backbone denoised {tuple(x.shape)} -> {tuple(out.shape)}")
    return True
  except Exception as e: return log("SWINDIT", e, False)

def demo_clip():
  from models.CLIP.clip import CLIP
  try:
    cfg = dict(embed_dim=512, image_resolution=224, vision_layers=12, vision_width=768, vision_patch_size=32,
                context_length=77, vocab_size=49408, transformer_width=512, transformer_heads=8, transformer_layers=12)
    model = CLIP(**cfg).to(DEVICE).eval()
    img, txt = torch.randn(1, 3, 224, 224, device=DEVICE), torch.randint(0, 49408, (1, 77), device=DEVICE)
    with torch.no_grad(): i_f = model.encode_image(img); t_f = model.encode_text(txt)
    log("CLIP", f"Encoded Image ({tuple(i_f.shape)}) and Text ({tuple(t_f.shape)})")
    return True
  except Exception as e: return log("CLIP", e, False)


if __name__ == "__main__":
  _DEMOS = {"vae": demo_vae, "diffusion": demo_diffusion, "swindit": demo_swindit, "clip": demo_clip}
  parser = argparse.ArgumentParser()
  parser.add_argument("--components", nargs="+", choices=_DEMOS.keys(), default=_DEMOS.keys())
  args = parser.parse_args()

  print(f"\033[1mAuth-SwinDiff Component Check\033[0m\nDevice: {DEVICE}\n{'-'*40}")
  results = {name: _DEMOS[name]() for name in args.components}
  
  passed = sum(filter(None, results.values()))
  print(f"\nStatus: {passed}/{len(args.components)} components ready.")