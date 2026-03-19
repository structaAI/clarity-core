import os
import json
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm

from src.datasets.auth_swin_dataset import AuthSwinDataset 
from src.models.swin_dit.backbone import SwinDiT
from src.utils.config_manager_swin_dit import load_config

# --- Paths ---
script_path = Path(__file__).resolve()
project_root = script_path.parents[2] 
load_dotenv(dotenv_path=project_root / ".env.local")

# --- Model Saving Directory ---
MODEL_SAVE_DIR = project_root / "src" / "models" / "swin_dit" / "saved_models"

def train():

  # 1. FORCE THE CORRECT GPU (Ensures Windows uses GPU 1, not GPU 0)
  os.environ["CUDA_VISIBLE_DEVICES"] = "0" # Usually PyTorch sees the RTX as 0 if Intel is ignored
  
  # 2. DISABLE HANG-PRONE OPTIMIZATIONS
  import torch.backends.cuda as cuda
  import torch.backends.cudnn as cudnn
  
  # These two help Blackwell GPUs handle Windows TDR (Timeouts)
  torch.backends.cuda.matmul.allow_tf32 = True 
  cudnn.benchmark = False
  # 1. Init Accelerator (BF16 for RTX 5070 Ti)
  accelerator = Accelerator(mixed_precision="bf16", project_dir="checkpoints/swindit")

  if accelerator.is_main_process:
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

  # 2. Setup
  config_path = project_root / "src" / "configs" / "swin_dit_config.yaml"
  swin_dit_config = load_config(str(config_path))

  model = SwinDiT(swin_dit_config)
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

  latent_dir = os.getenv("LATENT_CACHE_DIR")
  if not latent_dir:
    raise ValueError("LATENT_CACHE_DIR is not set in your .env.local file.")

  dataset = AuthSwinDataset(latent_dir)
  train_loader = DataLoader(
    dataset, 
    batch_size=8, 
    shuffle=True, 
    pin_memory=True, 
    num_workers=2, 
    persistent_workers=True, 
    multiprocessing_context='spawn' if os.name == 'nt' else None
  )

  # Prepare for CUDA
  model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

  # --- 3. RESUME LOGIC ---
  start_epoch = 0
  checkpoint_dir = Path("checkpoints/swindit/last")
  resume_file = checkpoint_dir / "resume_metadata.json"

  # Only attempt resume if the metadata file actually exists (avoiding empty folder crashes)
  if resume_file.exists():
    try:
      accelerator.load_state(str(checkpoint_dir))
      with open(resume_file, "r") as f:
        metadata = json.load(f)
        start_epoch = metadata.get("last_epoch", 0) + 1
      accelerator.print(f"Resumed from Checkpoint. Starting at Epoch {start_epoch}")
    except Exception as e:
      accelerator.print(f"Resume failed: {e}. Starting from scratch.")

  accelerator.print(f"Training on {accelerator.device}. Saving to {MODEL_SAVE_DIR}")

  # --- 4. TRAINING LOOP ---
  for epoch in range(start_epoch, 100):
    model.train()
    epoch_loss = 0

    progress_bar = tqdm(
      train_loader, 
      desc=f"Epoch {epoch}", 
      disable=not accelerator.is_local_main_process # Only show on 1 process
    )
    
    for batch in train_loader:
      optimizer.zero_grad()
      device = batch["lr"].device
      t = torch.zeros(batch["lr"].shape[0], device=device).long()
      
      outputs = model(batch["lr"], t)
      loss = F.mse_loss(outputs, batch["hr"])
      
      accelerator.backward(loss)
      torch.cuda.synchronize()
      optimizer.step()
      current_loss = loss.item()
      epoch_loss += current_loss

      progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

    # --- 5. PERIODIC SAVE LOGIC ---
    avg_loss = epoch_loss / len(train_loader)
    
    # Save Accelerator State for resuming
    accelerator.save_state(str(checkpoint_dir))
    
    # Save Model Weights for production
    if accelerator.is_main_process:
      unwrapped_model = accelerator.unwrap_model(model)
      save_path = MODEL_SAVE_DIR / f"swindit_epoch_{epoch}.pt"
      torch.save(unwrapped_model.state_dict(), save_path)
      
      # Atomic update of metadata
      with open(resume_file, "w") as f:
        json.dump({"last_epoch": epoch}, f)
    
    accelerator.print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Weights saved to {MODEL_SAVE_DIR.name}")

if __name__ == "__main__":
  train()