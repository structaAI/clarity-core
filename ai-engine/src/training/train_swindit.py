# --- Imports ---
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


"""
Training Loop for SwinDiT:

Dataset: 
- COCO image dataset (HR-LR pairs)
- Dataset Size: 11827 images (each)
- Total Epochs: 100

General
- Training Precision: (mixed)/bfloat16
- Initial Learning Rate: 1*e^-4
- Multiprocessing Type: Spawn; Default: nt
- Number of Workers: 2

Possible Issues
- Precision Loss since float32 was not used (Time and Memory Constraint)
- Excesive number of files saved due to Model Checkpointing
- Results may not be up to the mark and might need changes in either learning rate or number of epochs.
- Given that we are working with latent caches of images, i.e we have more than 3 dimensions, TQDM progress bar will have issues but
said issues do not hinder model training.

Future Validation Checks
- PSNR
- SSIM
- LPIPS

"""
def train():
  # Forcing a check on which GPU is being pointed at
  # Here our Nvidia GPU is set at GPU0; given that there are n-GPUs:
  #   {GPU0, GPU1, ...., GPUn-1}
  os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
  
  # 1. Cudann Benchmarks and Checks
  import torch.backends.cuda as cuda
  import torch.backends.cudnn as cudnn
  
  torch.backends.cuda.matmul.allow_tf32 = True 
  cudnn.benchmark = False

  # Setting precision to bfloat16 instead of float32 and setting the checkpoint directory
  accelerator = Accelerator(mixed_precision="bf16", project_dir="checkpoints/swindit")

  # Used to check for one process which is running in our GPU0
  # Save weights only from the main process to avoid conflicts and redundant saves in multi-GPU setups (Since we had GPU0 and GPU1)
  if accelerator.is_main_process:
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

  # 2. Setup
  config_path = project_root / "src" / "configs" / "swin_dit_config.yaml"
  swin_dit_config = load_config(str(config_path))

  # 3. Model, Optimizer, Dataset, and DataLoader Setup

  # Importing SwinDiT model
  model = SwinDiT(swin_dit_config)

  # Setting AdamW Optimizer
  optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

  # Checking for Latent Cache Directory --> Storing .safetensors files that we got after passing COCO images through VAE Encoder.
  latent_dir = os.getenv("LATENT_CACHE_DIR")
  if not latent_dir:
    raise ValueError("LATENT_CACHE_DIR is not set in your .env.local file.")

  # Loeading the dataset and setting up DataLoader
  dataset = AuthSwinDataset(latent_dir)
  train_loader = DataLoader(
    dataset, 
    batch_size=8, # Reduced from 32 to 8 (Speeding up Training)
    shuffle=True, 
    pin_memory=True, # Speeding up Data Transfer to GPU
    num_workers=2,  # Since we are having a smaller batch size and working with an RTX 5070ti,
    # Set num_workers = 0, if using regular windows w/o WSL;
    # can use num_workers = 4 for Linux or Windows with WSL (Since 2 physical cores and 4 logical cores)
    persistent_workers=True, 
    multiprocessing_context='spawn' if os.name == 'nt' else None
  )

  # Preparing Hugging Face Accelerator for mixed precision training and multi-GPU support
  model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

  # Pre-Training configurations
  start_epoch = 0
  checkpoint_dir = Path("checkpoints/swindit/last")
  resume_file = checkpoint_dir / "resume_metadata.json"

  # In case of break in training, check Resume File for last recorded epoch
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

  # 4. Main Training Loop 
  for epoch in range(start_epoch, 100):
    model.train()
    epoch_loss = 0

    progress_bar = tqdm(
      train_loader, 
      desc=f"Epoch {epoch}", 
      disable=not accelerator.is_local_main_process 
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

    # 5. Checkpointing and Saving Model Weights after every epoch
    avg_loss = epoch_loss / len(train_loader)
    
    # Save Accelerator State for resuming
    accelerator.save_state(str(checkpoint_dir))
    
    # Save Model Weights for production
    if accelerator.is_main_process:
      unwrapped_model = accelerator.unwrap_model(model)
      save_path = MODEL_SAVE_DIR / f"swindit_epoch_{epoch}.pt"
      torch.save(unwrapped_model.state_dict(), save_path)
      
      # Writing into Resume File to update Checkpoint
      with open(resume_file, "w") as f:
        json.dump({"last_epoch": epoch}, f)
    
    # Logging Epoch Loss and Sanity Check for every epoch
    accelerator.print(f"Epoch {epoch} | Loss: {avg_loss:.6f} | Weights saved to {MODEL_SAVE_DIR.name}")

# Driver Code
if __name__ == "__main__":
  train()