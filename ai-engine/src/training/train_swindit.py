"""
train_swindit.py
=================
Base training script for SwinDiT on latent SR pairs.

Changes from previous version
------------------------------
1. load_checkpoint updated to handle new architecture keys:
   - lr_patch_embedding.* (new secondary LR patch embed)
   - blocks.*.adaLN1.* / adaLN2.* (AdaLN replaces norm1/norm2)
   - blocks.*.rpb.* (relative position bias, new)
   - blocks.*.lr_cross_attn.* (LR cross-attention, new — only one block)
   - final_norm.* (new final LayerNorm before linear head)
   New keys are randomly initialised when loading from old checkpoints.

2. Everything else unchanged — the training loop, LR schedule, diffusion
   forward pass, and checkpoint strategy are identical.
"""

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

MODEL_SAVE_DIR        = project_root / "src" / "models" / "swin_dit" / "saved_models"
PRETRAINED_CHECKPOINT = os.getenv("PRETRAINED_CHECKPOINT", "")

# Keys introduced by the new architecture.
# When loading old checkpoints, these will be missing → randomly initialised.
_NEW_ARCH_PREFIXES = (
    "lr_patch_embedding.",
    "final_norm.",
    "blocks.",   # will be partially new (adaLN, rpb, lr_cross_attn)
)


def load_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """
    Universal checkpoint loader.

    Handles three cases:
      A) Old 4-ch patch_embedding → expand to 8-ch (legacy SR checkpoints)
      B) Old 8-ch checkpoint without new arch keys → load matching, init new
      C) Current checkpoint with all keys → direct strict load

    New architecture keys (AdaLN, RPB, LRCrossAttention, final_norm,
    lr_patch_embedding) are randomly initialised when absent.
    """
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]

    model_state = model.state_dict()
    loaded, skipped, newly_init = 0, [], []

    # Detect patch_embedding channel count
    proj_key        = "patch_embedding.projection.weight"
    ckpt_in_ch      = ckpt[proj_key].shape[1]  if proj_key in ckpt       else None
    model_in_ch     = model_state[proj_key].shape[1] if proj_key in model_state else None

    if ckpt_in_ch == 4 and model_in_ch == 8:
        print("  Mode: 4-ch → 8-ch expansion (HR weights copied, LR channels random-init)")
    elif ckpt_in_ch == 8 and model_in_ch == 8:
        print("  Mode: 8-ch checkpoint → loading matching keys, random-init new arch keys")
    else:
        print(f"  Mode: partial load (ckpt_in_ch={ckpt_in_ch}, model_in_ch={model_in_ch})")

    for k, v in ckpt.items():
        if k not in model_state:
            skipped.append(f"{k} (not in model)")
            continue

        if k == proj_key:
            if v.shape[1] == 4 and model_state[k].shape[1] == 8:
                new_w = torch.zeros_like(model_state[k])
                new_w[:, :4, :, :] = v
                nn.init.normal_(new_w[:, 4:, :, :], mean=0.0, std=0.02)
                model_state[k] = new_w
                loaded += 1
                print(f"  Expanded : {k}  {list(v.shape)} → {list(new_w.shape)}")
            elif v.shape == model_state[k].shape:
                model_state[k] = v
                loaded += 1
            else:
                skipped.append(f"{k} shape mismatch")
            continue

        if model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped.append(f"{k} shape {list(v.shape)} vs {list(model_state[k].shape)}")

    # Identify newly initialised keys (in model but not in ckpt)
    for k in model_state:
        if k not in ckpt:
            newly_init.append(k)

    model.load_state_dict(model_state, strict=False)
    print(f"  Loaded {loaded} tensors, skipped {len(skipped)}, "
          f"randomly-initialised {len(newly_init)} new arch keys.")
    if skipped:
        for s in skipped[:3]:
            print(f"    skip: {s}")
    if newly_init:
        # Print a sample so it's clear which new modules activated
        sample = [k for k in newly_init if any(
            p in k for p in ("lr_patch", "adaLN", "rpb", "lr_cross", "final_norm")
        )][:6]
        for k in sample:
            print(f"    new : {k}")
    print()
    return model


def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

    accelerator = Accelerator(mixed_precision="bf16", project_dir="checkpoints/swindit")

    if accelerator.is_main_process:
        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    config_path     = project_root / "src" / "configs" / "swin_dit_config.yaml"
    swin_dit_config = load_config(str(config_path))

    diffusion = GaussianDiffusion(
        num_timesteps=swin_dit_config.diffusion.num_sampling_steps,
        schedule=swin_dit_config.diffusion.noise_schedule,
    )

    model = SwinDiT(swin_dit_config)

    is_finetuning = bool(PRETRAINED_CHECKPOINT and Path(PRETRAINED_CHECKPOINT).exists())
    if is_finetuning:
        model = load_checkpoint(model, PRETRAINED_CHECKPOINT)
    else:
        accelerator.print("No pretrained checkpoint — training from scratch.")

    lr        = 3e-5 if is_finetuning else 1e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    accelerator.print(f"LR: {lr}  ({'fine-tuning' if is_finetuning else 'scratch'})")

    latent_dir = os.getenv("LATENT_CACHE_DIR")
    if not latent_dir:
        raise ValueError("LATENT_CACHE_DIR is not set in .env.local.")

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

    NUM_EPOCHS = 15 if is_finetuning else 30
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler)

    for attr in [
        "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        "sqrt_recip_alphas", "posterior_variance",
        "posterior_log_variance_clipped",
        "posterior_mean_coef1", "posterior_mean_coef2",
    ]:
        setattr(diffusion, attr, getattr(diffusion, attr).to(accelerator.device))

    start_epoch   = 0
    checkpoint_dir = project_root / "checkpoints" / "swindit" / "last"
    resume_file    = checkpoint_dir / "resume_metadata.json"

    if resume_file.exists():
        try:
            accelerator.load_state(str(checkpoint_dir))
            with open(resume_file) as f:
                start_epoch = json.load(f).get("last_epoch", 0) + 1
            accelerator.print(f"Resumed from checkpoint. Starting at Epoch {start_epoch}")
        except Exception as e:
            accelerator.print(f"Resume failed: {e}. Starting from scratch.")

    accelerator.print(
        f"Training on {accelerator.device}. "
        f"Dataset: {len(dataset)} pairs. "
        f"Epochs: {start_epoch} → {NUM_EPOCHS}"
    )

    T = swin_dit_config.diffusion.num_sampling_steps

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in pbar:
            optimizer.zero_grad()

            hr = batch["hr"]   # [B, 4, H, W]
            lr = batch["lr"]   # [B, 4, H, W]

            t     = torch.randint(0, T, (hr.shape[0],), device=hr.device, dtype=torch.long)
            noise = torch.randn_like(hr)
            x_t   = diffusion.q_sample(hr, t, noise=noise)

            # 8-channel input: noisy HR + LR conditioning
            # The backbone now internally extracts lr_tokens from channels 4:8
            x_input    = torch.cat([x_t, lr], dim=1)   # [B, 8, H, W]
            noise_pred = model(x_input, t)[:, :4, :, :]

            loss = F.mse_loss(noise_pred, noise)

            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            current_loss = loss.item()
            epoch_loss  += current_loss
            pbar.set_postfix({"loss": f"{current_loss:.4f}"})

        scheduler.step()

        avg_loss   = epoch_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[0]

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save_state(str(checkpoint_dir))

        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            save_path = MODEL_SAVE_DIR / f"swindit_v2_epoch_{epoch+1}.pt"
            torch.save(unwrapped.state_dict(), save_path)
            with open(resume_file, "w") as f:
                json.dump({"last_epoch": epoch}, f)

        accelerator.print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Loss: {avg_loss:.6f} | "
            f"LR: {current_lr:.2e} | "
            f"Saved → {MODEL_SAVE_DIR.name}/swindit_v2_epoch_{epoch+1}.pt"
        )


if __name__ == "__main__":
    train()