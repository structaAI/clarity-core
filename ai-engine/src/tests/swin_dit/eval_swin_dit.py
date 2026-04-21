import torch
import time
import sys
from pathlib import Path
from types import SimpleNamespace

current_file = Path(__file__).resolve()
src_path = current_file.parents[2] 
if str(src_path) not in sys.path:
  sys.path.insert(0, str(src_path))

# Now use absolute imports instead of relative
from models.swin_dit.backbone import SwinDiT

def evaluate_swindit_efficiency(model, latent_size=64):
  model.to("cuda").eval()
  

  dummy_input = torch.randn(1, 4, latent_size, latent_size).cuda()
  dummy_timesteps = torch.tensor([500]).cuda()
  with torch.no_grad():
    _ = model(dummy_input, dummy_timesteps)
  

  torch.cuda.synchronize()
  t0 = time.perf_counter()
  
  for _ in range(100):
    with torch.no_grad():
      _ = model(dummy_input, dummy_timesteps)
          
  torch.cuda.synchronize()
  t1 = time.perf_counter()
  
  iters_per_sec = 100 / (t1 - t0)
  
  # 2. VRAM Peak (Crucial for 5070 Ti)
  mem = torch.cuda.max_memory_allocated() / (1024**2)
  
  print(f"\n\033[1mSwin-DiT Efficiency Report\033[0m")
  print(f"{'-'*30}")
  print(f"Throughput:     {iters_per_sec:.2f} it/s")
  print(f"Peak VRAM:       {mem:.2f} MB")
  print(f"Device:         {torch.cuda.get_device_name(0)}")

if __name__ == "__main__":
  cfg = SimpleNamespace(model=SimpleNamespace(
    latent_size=64, 
    in_channels=4, 
    patch_size=2, 
    embed_dim=768, 
    depths=[2,2,2,2], 
    num_heads=[3,6,12,24], 
    window_size=8, 
    use_pswa_bridge=True
  ))
  
  model = SwinDiT(cfg)
  evaluate_swindit_efficiency(model)