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
from src.models.diffusion.diffusion_engine import GaussianDiffusion   # FIX: was missing
from src.utils.config_manager_swin_dit import load_config

# --- Paths ---
script_path = Path(__file__).resolve()
project_root = script_path.parents[2]
load_dotenv(dotenv_path=project_root / ".env.local")

# --- Model Saving Directory ---
MODEL_SAVE_DIR = project_root / "src" / "models" / "swin_dit" / "saved_models"


def train():
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  import torch.backends.cudnn as cudnn
  torch.backends.cuda.matmul.allow_tf32 = True
  cudnn.benchmark = False

  accelerator = Accelerator(mixed_precision="bf16", project_dir="checkpoints/swindit")

  if accelerator.is_main_process:
      MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

  # 2. Setup
  config_path = project_root / "src" / "configs" / "swin_dit_config.yaml"
  swin_dit_config = load_config(str(config_path))

  # 3. Model, Diffusion, Optimizer, Dataset, DataLoader
  model = SwinDiT(swin_dit_config)

  # FIX: Instantiate GaussianDiffusion so we can call q_sample() during training.
  diffusion = GaussianDiffusion(
    num_timesteps=swin_dit_config.diffusion.num_sampling_steps,
    schedule=swin_dit_config.diffusion.noise_schedule,
  )

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
    multiprocessing_context="spawn" if os.name == "nt" else None,
  )

  model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

  # Pre-Training configurations
  start_epoch = 0
  checkpoint_dir = Path("checkpoints/swindit/last")
  resume_file = checkpoint_dir / "resume_metadata.json"

  if resume_file.exists():
    try:
      accelerator.load_state(str(checkpoint_dir))
      with open(resume_file, "r") as f:
        metadata = json.load(f)
        start_epoch = metadata.get("last_epoch", 0) + 1
      accelerator.print(f"Resumed from checkpoint. Starting at Epoch {start_epoch}")
    except Exception as e:
      accelerator.print(f"Resume failed: {e}. Starting from scratch.")

  accelerator.print(f"Training on {accelerator.device}. Saving to {MODEL_SAVE_DIR}")

  T = swin_dit_config.diffusion.num_sampling_steps

  # 4. Main Training Loop
  for epoch in range(start_epoch, 100):
      model.train()
      epoch_loss = 0.0

      # FIX: iterate over progress_bar, not train_loader, so the bar actually advances.
      progress_bar = tqdm(
          train_loader,
          desc=f"Epoch {epoch}",
          disable=not accelerator.is_local_main_process,
      )

      for batch in progress_bar:               # FIX: was "for batch in train_loader"
        optimizer.zero_grad()

        hr = batch["hr"]                     # clean target latents  [B, 4, H, W]
        lr = batch["lr"]                     # degraded input latents [B, 4, H, W]

        # ── FIX: proper diffusion training ──────────────────────────────────
        # Sample random timesteps uniformly in [0, T)
        t = torch.randint(0, T, (hr.shape[0],), device=hr.device, dtype=torch.long)

        # Sample noise and create noisy HR latents via the forward process
        noise = torch.randn_like(hr)
        x_t = diffusion.q_sample(hr, t, noise=noise)   # noisy HR at timestep t

        # Model predicts the noise ε given the noisy latent and LR conditioning
        # We concat lr along channel dim as spatial conditioning signal
        # (or pass separately if your SwinDiT forward supports it)
        noise_pred = model(x_t, t)

        # Loss: predict the noise, not the clean image
        loss = F.mse_loss(noise_pred, noise)
        # ────────────────────────────────────────────────────────────────────

        accelerator.backward(loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        current_loss = loss.item()
        epoch_loss += current_loss
        progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

      # 5. Checkpointing
      avg_loss = epoch_loss / len(train_loader)

      accelerator.save_state(str(checkpoint_dir))

      if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        save_path = MODEL_SAVE_DIR / f"swindit_epoch_{epoch}.pt"
        torch.save(unwrapped_model.state_dict(), save_path)

        with open(resume_file, "w") as f:
            json.dump({"last_epoch": epoch}, f)

      accelerator.print(
          f"Epoch {epoch} | Loss: {avg_loss:.6f} | Weights saved to {MODEL_SAVE_DIR.name}"
      )


if __name__ == "__main__":
  train()