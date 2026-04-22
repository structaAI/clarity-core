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


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """
    Universal checkpoint loader — handles both:
      - Old 4-channel checkpoints  → expands patch_embedding to 8ch
      - Current 8-channel checkpoints → direct load, no expansion

    Detects which case applies by inspecting patch_embedding.projection.weight
    shape in the checkpoint before doing anything.
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Unwrap packaged checkpoints
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    model_state = model.state_dict()
    loaded, skipped = 0, []

    # Detect checkpoint channel count before iterating
    proj_key = "patch_embedding.projection.weight"
    ckpt_in_channels = ckpt[proj_key].shape[1] if proj_key in ckpt else None
    model_in_channels = model_state[proj_key].shape[1] if proj_key in model_state else None

    if ckpt_in_channels is not None and model_in_channels is not None:
        if ckpt_in_channels == 4 and model_in_channels == 8:
            print(f"  Mode: 4-ch → 8-ch expansion  "
                  f"(HR weights copied, LR channels randomly initialised)")
        elif ckpt_in_channels == 8 and model_in_channels == 8:
            print(f"  Mode: direct load  (checkpoint already 8-ch, no expansion needed)")
        else:
            print(f"  Mode: shape mismatch  "
                  f"(ckpt={ckpt_in_channels}ch, model={model_in_channels}ch) — will skip mismatches")

    for k, v in ckpt.items():
        if k not in model_state:
            skipped.append(f"{k} (not in model)")
            continue

        if k == proj_key:
            if v.shape[1] == 4 and model_state[k].shape[1] == 8:
                # Old 4-ch checkpoint: expand to 8ch
                # Channels 0-3: copy trained HR weights
                # Channels 4-7: small random init so LR conditioning activates gradually
                new_w = torch.zeros_like(model_state[k])
                new_w[:, :4, :, :] = v
                nn.init.normal_(new_w[:, 4:, :, :], mean=0.0, std=0.02)
                model_state[k] = new_w
                loaded += 1
                print(f"  Expanded : {k}  {list(v.shape)} → {list(new_w.shape)}")
            elif v.shape == model_state[k].shape:
                # Already 8-ch: direct copy
                model_state[k] = v
                loaded += 1
                print(f"  Loaded   : {k}  {list(v.shape)}  (already 8-ch)")
            else:
                skipped.append(
                    f"{k} (unexpected shape: ckpt={list(v.shape)} "
                    f"model={list(model_state[k].shape)})"
                )
            continue

        # All other keys: load if shapes match
        if model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped.append(
                f"{k} (shape mismatch: ckpt={list(v.shape)} "
                f"model={list(model_state[k].shape)})"
            )

    model.load_state_dict(model_state)
    print(f"  Loaded {loaded}/{len(model_state)} tensors, skipped {len(skipped)}.")
    if skipped:
        for s in skipped:
            print(f"    skip: {s}")
    print()
    return model


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

    # Load pretrained checkpoint if provided
    is_finetuning = bool(PRETRAINED_CHECKPOINT and Path(PRETRAINED_CHECKPOINT).exists())
    if is_finetuning:
        model = load_checkpoint(model, PRETRAINED_CHECKPOINT)
    else:
        accelerator.print("No pretrained checkpoint found — training from scratch.")

    # --- Optimizer ---
    # Lower LR for fine-tuning (weights already trained), full LR for scratch
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

    # --- LR scheduler: cosine decay over NUM_EPOCHS ---
    NUM_EPOCHS = 15 if is_finetuning else 30
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
        eta_min=1e-6,
    )

    # --- Accelerate preparation ---
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # --- Move diffusion buffers to device to avoid fp32 promotion ---
    for attr in [
        "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        "sqrt_recip_alphas", "posterior_variance",
        "posterior_log_variance_clipped",
        "posterior_mean_coef1", "posterior_mean_coef2",
    ]:
        setattr(diffusion, attr, getattr(diffusion, attr).to(accelerator.device))

    # --- Resume logic ---
    start_epoch = 0
    checkpoint_dir = project_root / "checkpoints" / "swindit" / "last"
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
    accelerator.print(f"Epochs: {start_epoch} → {NUM_EPOCHS}")

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
            lr = batch["lr"]   # LR latent upsampled [B, 4, H, W]

            # Sample timesteps and add noise to HR latent
            t = torch.randint(0, T, (hr.shape[0],), device=hr.device, dtype=torch.long)
            noise = torch.randn_like(hr)
            x_t = diffusion.q_sample(hr, t, noise=noise)   # [B, 4, H, W]

            # Concatenate noisy HR + LR along channel dim for SR conditioning
            x_input = torch.cat([x_t, lr], dim=1)           # [B, 8, H, W]

            # Model predicts noise — slice first 4 channels for HR supervision
            noise_pred = model(x_input, t)[:, :4, :, :]     # [B, 4, H, W]

            loss = F.mse_loss(noise_pred, noise)

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
            save_path = MODEL_SAVE_DIR / f"swindit_v2_epoch_{epoch + 1}.pt"
            torch.save(unwrapped_model.state_dict(), save_path)

            with open(resume_file, "w") as f:
                json.dump({"last_epoch": epoch}, f)

        accelerator.print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Loss: {avg_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Saved → {MODEL_SAVE_DIR.name}/swindit_v2_epoch_{epoch + 1}.pt"
        )


if __name__ == "__main__":
    train()