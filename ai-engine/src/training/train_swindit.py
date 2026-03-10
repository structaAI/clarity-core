"""
Path: src/training/train_swindit.py
Main training script with checkpointing and BF16 support.
"""
import os
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets.auth_swin_dataset import AuthSwinDataset

from models.swin_dit.backbone import SwinDiT
from utils.config_manager_swin_dit import load_config
  
swin_dit_config = load_config("configs/swin_dit_config.yaml")

def train():
  # 1. Init Accelerator (Handles BF16 and Checkpoints)
  accelerator = Accelerator(mixed_precision="bf16", project_dir="checkpoints/swindit")
  
  # 2. Setup Model & Optimizer
  # Note: Ensure your config is defined or passed here
  model = SwinDiT(swin_dit_config)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
  
  # 3. Data
  dataset = AuthSwinDataset(os.getenv("LATENT_CACHE_DIR", "data/latents"))
  train_loader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True)

  # 4. Prepare for 5070 Ti
  model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

  # 5. Resume Logic
  if os.path.exists("checkpoints/swindit/last"):
    accelerator.load_state("checkpoints/swindit/last")
    accelerator.print("Checkpoint loaded. Resuming...")

  # 6. Training Loop
  for epoch in range(100):
    model.train()
    for batch in train_loader:
      optimizer.zero_grad()
      outputs = model(batch["lr"])
      loss = F.mse_loss(outputs, batch["hr"])
      accelerator.backward(loss)
      optimizer.step()

    # 7. Save State
    accelerator.save_state("checkpoints/swindit/last")
    accelerator.print(f"Epoch {epoch} complete. Saved checkpoint.")

if __name__ == "__main__":
  train()