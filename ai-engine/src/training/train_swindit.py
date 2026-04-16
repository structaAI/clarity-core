# --- Imports ---
import sys
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm

script_path = Path(__file__).resolve()
project_root = script_path.parents[2]

# Bootstrap: add project root to path so src.* imports resolve
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env.local")

from src.datasets.auth_swin_dataset import AuthSwinDataset
from src.models.swin_dit.backbone import SwinDiT
from src.models.diffusion.diffusion_engine import GaussianDiffusion
from src.utils.config_manager_swin_dit import load_config

# --- Model Saving Directory ---
MODEL_SAVE_DIR = project_root / "src" / "models" / "swin_dit" / "saved_models"

# --- Checkpoint to fine-tune from (set to "" to train from scratch) ---
PRETRAINED_CHECKPOINT = os.getenv("PRETRAINED_CHECKPOINT", "")


# ---------------------------------------------------------------------------
# Checkpoint loader — expands patch_embedding from 4 → 8 channels for LR
# concatenation, loads all other layers directly where shapes match.
# final_layer is intentionally skipped — it must reinitialise because
# in_channels changed from 4 → 8, so output channels doubled (16 → 32).
# ---------------------------------------------------------------------------

def load_checkpoint_with_channel_expansion(model: nn.Module, checkpoint_path: str) -> nn.Module:
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Handle both raw state dicts and packaged checkpoints
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    model_state = model.state_dict()
    loaded, skipped = 0, []

    for k, v in ckpt.items():
        if k not in model_state:
            skipped.append(f"{k} (not in model)")
            continue

        # Special case: expand patch embedding 4 → 8 input channels
        if k == "patch_embedding.projection.weight":
            # Checkpoint: [768, 4, 2, 2]   Target: [768, 8, 2, 2]
            new_weight = torch.zeros_like(model_state[k])
            new_weight[:, :4, :, :] = v                                   # HR channels — trained weights
            nn.init.normal_(new_weight[:, 4:, :, :], mean=0.0, std=0.02)  # LR channels — fresh init
            model_state[k] = new_weight
            loaded += 1
            print(f"  Expanded : {k}  {list(v.shape)} → {list(new_weight.shape)}")
            continue

        # Skip final_layer — output channels changed (16 → 32) so it must reinitialise
        if k.startswith("final_layer"):
            skipped.append(f"{k} (intentionally skipped — output channels changed, will reinitialise)")
            continue

        if model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped.append(f"{k} (shape mismatch: ckpt={list(v.shape)} model={list(model_state[k].shape)})")

    model.load_state_dict(model_state)
    print(f"  Loaded : {loaded}/{len(model_state)} tensors")
    if skipped:
        print(f"  Skipped: {len(skipped)}")
        for s in skipped:
            print(f"    {s}")
    print()
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    accelerator = Accelerator(mixed_precision="bf16", project_dir="checkpoints/swindit")

    if accelerator.is_main_process:
        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Config ---
    config_path = project_root / "src" / "configs" / "swin_dit_config.yaml"
    swin_dit_config = load_config(str(config_path))

    # --- Diffusion scheduler ---
    diffusion = GaussianDiffusion(
        num_timesteps=swin_dit_config.diffusion.num_sampling_steps,
        schedule=swin_dit_config.diffusion.noise_schedule,
    )

    # --- Model ---
    model = SwinDiT(swin_dit_config)

    # Load pretrained checkpoint if provided, else train from scratch
    is_finetuning = bool(PRETRAINED_CHECKPOINT and Path(PRETRAINED_CHECKPOINT).exists())
    if is_finetuning:
        model = load_checkpoint_with_channel_expansion(model, PRETRAINED_CHECKPOINT)
    else:
        accelerator.print("No pretrained checkpoint found — training from scratch.")

    # --- Optimizer ---
    # Lower lr for fine-tuning (HR weights already trained), full lr for scratch
    lr = 3e-5 if is_finetuning else 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    accelerator.print(f"Learning rate: {lr}  ({'fine-tuning' if is_finetuning else 'scratch'})")

    # --- Dataset & DataLoader ---
    latent_dir = os.getenv("LATENT_CACHE_DIR")
    if not latent_dir:
        raise ValueError("LATENT_CACHE_DIR is not set in your .env.local file.")

    dataset = AuthSwinDataset(latent_dir)
    train_loader = DataLoader(
        dataset,
        batch_size=swin_dit_config.training.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="spawn" if os.name == "nt" else None,
    )

    # --- LR scheduler: cosine decay over 30 epochs ---
    NUM_EPOCHS = 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6,
    )

    # --- Accelerate preparation ---
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # --- Resume logic ---
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
    accelerator.print(f"Dataset size: {len(dataset)} latent pairs")
    accelerator.print(f"Steps per epoch: {len(train_loader)}")

    T = swin_dit_config.diffusion.num_sampling_steps

    # --- Main Training Loop ---
    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in progress_bar:
            optimizer.zero_grad()

            hr = batch["hr"]   # clean HR latent    [B, 4, H, W]
            lr = batch["lr"]   # degraded LR latent [B, 4, H, W]

            # Sample timesteps and add noise to HR latent via proper DDPM forward process
            t = torch.randint(0, T, (hr.shape[0],), device=hr.device, dtype=torch.long)
            noise = torch.randn_like(hr)
            x_t = diffusion.q_sample(hr, t, noise=noise)   # [B, 4, H, W]

            # Concatenate noisy HR + LR along channel dim for SR conditioning
            x_input = torch.cat([x_t, lr], dim=1)           # [B, 8, H, W]

            # Model predicts noise — output is 8 channels because in_channels=8
            # Slice first 4 channels only — we only supervise the HR noise prediction
            noise_pred = model(x_input, t)[:, :4, :, :]     # [B, 4, H, W]

            # Loss: MSE between predicted and actual noise (epsilon prediction)
            loss = F.mse_loss(noise_pred, noise)             # both [B, 4, H, W]

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_loss = loss.item()
            epoch_loss += current_loss
            progress_bar.set_postfix({"loss": f"{current_loss:.4f}"})

        # Step LR scheduler once per epoch
        scheduler.step()

        avg_loss = epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        # --- Checkpointing ---
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(str(checkpoint_dir))

        if accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            save_path = MODEL_SAVE_DIR / f"swindit_epoch_{epoch + 1}.pt"
            torch.save(unwrapped_model.state_dict(), save_path)

            with open(resume_file, "w") as f:
                json.dump({"last_epoch": epoch}, f)

        accelerator.print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Loss: {avg_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Saved → {MODEL_SAVE_DIR.name}/swindit_epoch_{epoch + 1}.pt"
        )


if __name__ == "__main__":
    train()