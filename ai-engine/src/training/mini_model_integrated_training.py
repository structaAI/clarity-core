import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import GradScaler
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
import os
import yaml
from types import SimpleNamespace
from typing import Any, Dict

# Internal Imports
from src.datasets.coco_restoration import CocoAuthDataset
from src.models.bridge.auth_bridge import AuthBridge
from src.models.swin_dit.backbone import SwinDiT

# 1. Config Loader
def load_config(config_path: str) -> SimpleNamespace:
  with open(config_path, 'r') as file:
    config_dict = yaml.safe_load(file)
  
  def dict_to_namespace(d: Any) -> Any:
    if isinstance(d, dict):
      return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    return d
  return dict_to_namespace(config_dict)

def run_overfit_training():
  device = torch.device("cuda")
  config = load_config("src/configs/model_config.yaml")

  full_dataset = CocoAuthDataset(
    img_dir=os.getenv("COCO_IMG_PATH", "./data/train2017"),
    ann_file=os.getenv("COCO_ANN_PATH", "./data/annotations/captions_train2017.json")
  )
  indices = list(range(100))
  subset_dataset = Subset(full_dataset, indices)
  train_loader = DataLoader(subset_dataset, batch_size=config.training.batch_size, shuffle=True)

  bridge = AuthBridge(input_dim=1152, output_dim=config.model.embed_dim).to(device)
  
  model = SwinDiT(config.model).to(device) 
  
  optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(bridge.parameters()), 
    lr=config.training.learning_rate
  )
  scaler = GradScaler()

  print(f"Starting mini-model training on {device}...")
  model.train()
  bridge.train()

  for epoch in range(300): # Increased to 300 for perfect memorization
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    
    for batch in pbar:
      optimizer.zero_grad()
      
      # 4. Prepare Inputs
      # In a real run, these should be pre-encoded latents to save VRAM
      # For this overfit, we'll assume the dataset provides what's needed
      hr_latents = batch['hr'].to(device)
      lr_latents = batch['lr'].to(device)
      clip_emb = batch['clip_emb'].to(device)
      
      # Timestep for Demo (Diffusion-style training)
      t = torch.randint(0, 1000, (hr_latents.shape[0],), device=device).long()

      # Mixed Precision for Blackwell Speed
      with autocast(device_type='cuda'):
        cond = bridge(
          t_emb=t, 
          clip_emb=clip_emb, 
          deg_type=batch['deg_id'].to(device),
          severity=batch['severity'].to(device)
        )
        
        # RESTORE: LR -> HR
        output = model(lr_latents, cond)
        
        loss = F.mse_loss(output, hr_latents)

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      
      total_loss += loss.item()
      pbar.set_postfix({"loss": f"{loss.item():.6f}"})

    # Save Checkpoint and Log
    if epoch % 10 == 0:
      os.makedirs("experiments", exist_ok=True)
      torch.save({
        'model_state_dict': model.state_dict(),
        'bridge_state_dict': bridge.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
      }, "experiments/auth_swindiff_demo.pt")

    print("Demo Mini Model Complete")

if __name__ == "__main__":
  run_overfit_training()