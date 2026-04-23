"""
train_perceptual.py
====================
Fine-tunes SwinDiT with a combined MSE + perceptual loss.

Changes from previous version
------------------------------
1. LAMBDA_PERC default raised to 0.05 (was 0.001 — too small to steer gradients
   against VGG feature MSE which is naturally in the 40-150 range on 512px images).

2. Perceptual loss spatial normalisation.
   VGG feature MSE scales with the spatial resolution of the decoded image
   (512px inputs produce much larger feature activations than 128px inputs).
   Dividing by (H * W) of the latent makes lambda_perc resolution-invariant,
   so the chosen value is stable regardless of HR_SIZE.

3. LEARNING_RATE default raised to 3e-5 (was 1e-5 — insufficient to move
   weights at the fine-tuning stage, resulting in flat loss curves).

4. F32 double-cast removed from _decode_latent (was casting to float32 twice).

5. Loss weighting log added to epoch summary for diagnostic visibility.

Why this matters
----------------
Pure epsilon-MSE trains the model to predict the MEAN of all plausible
HR images consistent with the LR input. The mean of many sharp textures
is a blurry texture. This is the fundamental ceiling of MSE-only training.

Adding a perceptual loss on the predicted x0 gives the model a direct
signal: "the reconstructed image should have sharp, perceptually realistic
features". This is how SR3, StableSR, and Real-ESRGAN achieve visual sharpness.

Loss formulation
----------------
  pred_x0  = (x_t - sqrt(1 - alpha_t) * eps_pred) / sqrt(alpha_t)
  L_mse    = MSE(eps_pred, eps)
  L_perc   = LPIPS_vgg(pred_x0_decoded, x0_decoded) / (H * W)
  L_total  = L_mse + lambda_perc * L_perc

The perceptual loss is only applied at LOW NOISE timesteps (t < PERC_T_MAX = 250)
where pred_x0 is meaningful. At high noise, pred_x0 is too corrupted to give a
useful perceptual signal and including it destabilises training.

Usage
-----
  python src/training/train_perceptual.py

Required .env.local:
  PRETRAINED_CHECKPOINT   — swindit_perc_epoch_N.pt or swindit_v2_epoch_8.pt
  LATENT_CACHE_DIR        — directory of .safetensors latent pairs
  VAE_PATH                — local AutoencoderKL directory

Optional .env.local:
  LEARNING_RATE=3e-5      (default)
  LAMBDA_PERC=0.05        (default — reduce to 0.02 if perc > 15 after epoch 1)
  PERC_T_MAX=250          (default)
  NUM_EPOCHS=10           (default)
  BATCH_SIZE=4            (default)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# ── Bootstrap ─────────────────────────────────────────────────────
script_path  = Path(__file__).resolve()
project_root = script_path.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
load_dotenv(dotenv_path=project_root / ".env.local")

from src.datasets.auth_swin_dataset import AuthSwinDataset
from src.models.swin_dit.backbone import SwinDiT
from src.models.diffusion.diffusion_engine import GaussianDiffusion
from src.utils.config_manager_swin_dit import load_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────
PRETRAINED_CHECKPOINT = os.getenv("PRETRAINED_CHECKPOINT", "")
LATENT_CACHE_DIR      = os.getenv("LATENT_CACHE_DIR", "")
VAE_PATH              = os.getenv("VAE_PATH", "")
NUM_EPOCHS            = int(os.getenv("NUM_EPOCHS",    "10"))
BATCH_SIZE            = int(os.getenv("BATCH_SIZE",    "4"))
LEARNING_RATE         = float(os.getenv("LEARNING_RATE", "3e-5"))   # FIX: was 1e-5
LAMBDA_PERC           = float(os.getenv("LAMBDA_PERC",  "0.05"))    # FIX: was 0.001
PERC_T_MAX            = int(os.getenv("PERC_T_MAX",   "250"))
GRAD_CLIP             = float(os.getenv("GRAD_CLIP",   "1.0"))
MODEL_SAVE_DIR        = project_root / "src" / "models" / "swin_dit" / "saved_models"
CHECKPOINT_DIR        = project_root / "checkpoints" / "perceptual" / "last"


# ── Perceptual Loss (VGG) ─────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    VGG-16 perceptual loss using relu2_2 and relu3_3 features.

    The model is frozen — only used as a fixed feature extractor.
    Input images must be decoded pixel tensors in [0, 1] range (3-channel RGB).
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        import torchvision.models as tvm
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:10]).to(device).eval()
        self.slice2 = nn.Sequential(*list(vgg.children())[10:17]).to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)

        # ImageNet normalisation
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target : [B, 3, H, W]  pixel images in [0, 1].
        Returns scalar perceptual loss (NOT spatially normalised — caller divides by H*W).
        """
        pred   = (pred   - self.mean) / self.std   # type: ignore[operator]
        target = (target - self.mean) / self.std    # type: ignore[operator]

        p1, t1 = self.slice1(pred),  self.slice1(target)
        p2, t2 = self.slice2(p1),    self.slice2(t1)

        return F.mse_loss(p1, t1) + F.mse_loss(p2, t2)


# ── Checkpoint loader ─────────────────────────────────────────────

def load_checkpoint(model: nn.Module, path: str) -> nn.Module:
    log.info(f"Loading checkpoint: {path}")
    ckpt  = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)

    model_state = model.state_dict()
    loaded, skipped = 0, []

    for k, v in state.items():
        if k not in model_state:
            skipped.append(f"{k} (not in model)")
            continue
        if model_state[k].shape == v.shape:
            model_state[k] = v
            loaded += 1
        else:
            skipped.append(f"{k} shape mismatch: ckpt={list(v.shape)} model={list(model_state[k].shape)}")

    model.load_state_dict(model_state, strict=False)
    log.info(f"  Loaded {loaded}/{len(model_state)} tensors.")
    if skipped:
        for s in skipped[:5]:
            log.warning(f"  skip: {s}")
        if len(skipped) > 5:
            log.warning(f"  ... and {len(skipped) - 5} more skipped.")
    return model


# ── Training ──────────────────────────────────────────────────────

def train() -> None:
    accelerator = Accelerator(
        mixed_precision="bf16",
        project_dir=str(project_root / "checkpoints" / "perceptual"),
    )

    if accelerator.is_main_process:
        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Config
    swin_config = load_config(
        str(project_root / "src" / "configs" / "swin_dit_config.yaml"))
    T_train = swin_config.diffusion.num_sampling_steps

    # Diffusion
    diffusion = GaussianDiffusion(
        num_timesteps=T_train,
        schedule=swin_config.diffusion.noise_schedule,
    )

    # Model
    model = SwinDiT(swin_config)
    if PRETRAINED_CHECKPOINT and Path(PRETRAINED_CHECKPOINT).exists():
        model = load_checkpoint(model, PRETRAINED_CHECKPOINT)
        accelerator.print(f"Fine-tuning from: {PRETRAINED_CHECKPOINT}")
    else:
        accelerator.print("WARNING: no checkpoint — training from scratch.")

    # VAE (frozen, float32 for decode stability)
    if not VAE_PATH or not Path(VAE_PATH).exists():
        raise FileNotFoundError(
            f"VAE_PATH not found: '{VAE_PATH}'. Set VAE_PATH in .env.local."
        )
    from diffusers import AutoencoderKL  # type: ignore
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True)
    vae = vae.to(accelerator.device, torch.float32).eval()  # type: ignore
    for p in vae.parameters():
        p.requires_grad_(False)
    VAE_SCALE = 0.13025

    # Perceptual loss
    perc_loss_fn = PerceptualLoss(device=accelerator.device)

    # Dataset
    if not LATENT_CACHE_DIR:
        raise ValueError("LATENT_CACHE_DIR is not set in .env.local.")
    dataset      = AuthSwinDataset(LATENT_CACHE_DIR)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="spawn" if os.name == "nt" else None,
    )

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler)

    # Move diffusion buffers to device
    for attr in [
        "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        "sqrt_recip_alphas", "posterior_variance",
        "posterior_log_variance_clipped",
        "posterior_mean_coef1", "posterior_mean_coef2",
    ]:
        setattr(diffusion, attr, getattr(diffusion, attr).to(accelerator.device))

    # Resume
    resume_file = CHECKPOINT_DIR / "resume_metadata.json"
    start_epoch = 0
    if resume_file.exists():
        try:
            accelerator.load_state(str(CHECKPOINT_DIR))
            with open(resume_file) as f:
                start_epoch = json.load(f).get("last_epoch", 0) + 1
            accelerator.print(f"Resumed from epoch {start_epoch}")
        except Exception as e:
            accelerator.print(f"Resume failed ({e}). Starting from epoch 0.")

    accelerator.print(
        f"\n{'='*62}\n"
        f"  Perceptual Fine-Tuning\n"
        f"  Device        : {accelerator.device}\n"
        f"  Epochs        : {start_epoch} → {NUM_EPOCHS}\n"
        f"  LR            : {LEARNING_RATE}\n"
        f"  lambda_perc   : {LAMBDA_PERC}\n"
        f"  Perc t_max    : t < {PERC_T_MAX} (low-noise steps only)\n"
        f"  Dataset       : {len(dataset)} pairs\n"
        f"  Note: if perc > 15 after epoch 1, reduce LAMBDA_PERC to 0.02\n"
        f"{'='*62}"
    )

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()

        epoch_mse  = 0.0
        epoch_perc = 0.0
        epoch_tot  = 0.0
        perc_steps = 0
        n_batches  = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in pbar:
            optimizer.zero_grad()

            hr = batch["hr"]   # [B, 4, H, W]
            lr = batch["lr"]   # [B, 4, H, W]
            B  = hr.shape[0]

            # Forward diffusion
            t     = torch.randint(0, T_train, (B,),
                                  device=hr.device, dtype=torch.long)
            noise = torch.randn_like(hr)
            x_t   = diffusion.q_sample(hr, t, noise=noise)

            x_in  = torch.cat([x_t, lr], dim=1)   # [B, 8, H, W]

            # Model forward — backbone now threads lr_tokens internally
            eps_pred = model(x_in, t)[:, :4]       # [B, 4, H, W]

            # MSE loss
            loss_mse = F.mse_loss(eps_pred, noise)

            # Perceptual loss (low-noise steps only)
            loss_perc = torch.tensor(0.0, device=hr.device, dtype=hr.dtype)
            low_noise_mask = t < PERC_T_MAX

            if low_noise_mask.any() and LAMBDA_PERC > 0:
                idx = low_noise_mask.nonzero(as_tuple=True)[0]

                # Reconstruct pred_x0
                at   = diffusion._extract(
                    diffusion.sqrt_alphas_cumprod, t[idx], hr[idx].shape)
                s1mt = diffusion._extract(
                    diffusion.sqrt_one_minus_alphas_cumprod, t[idx], hr[idx].shape)

                pred_x0 = (x_t[idx] - s1mt * eps_pred[idx]) / at.clamp(min=1e-3)
                pred_x0 = pred_x0.clamp(-1.0, 1.0)

                # Decode through VAE in float32 (single cast — FIX: removed redundant cast)
                with torch.no_grad():
                    def _decode(z: torch.Tensor) -> torch.Tensor:
                        decoded = vae.decode(z.float() / VAE_SCALE).sample  # type: ignore
                        return ((decoded.clamp(-1, 1) + 1.0) / 2.0)  # [0, 1]

                    pred_pixel = _decode(pred_x0)   # [B', 3, H*8, W*8]
                    gt_pixel   = _decode(hr[idx])

                # Perceptual loss with spatial normalisation (FIX: resolves scale issue)
                # VGG feature MSE is naturally ~40-150 on 512px images.
                # Dividing by latent spatial area (H*W) makes lambda_perc
                # resolution-invariant and numerically comparable to MSE.
                H_lat, W_lat = hr.shape[-2], hr.shape[-1]
                raw_perc = perc_loss_fn(pred_pixel.float(), gt_pixel.float())
                loss_perc = (raw_perc / (H_lat * W_lat)).to(hr.dtype)

                perc_steps += 1

            # Combined loss
            loss = loss_mse + LAMBDA_PERC * loss_perc

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            mse_v  = loss_mse.item()
            perc_v = loss_perc.item()
            tot_v  = loss.item()
            epoch_mse  += mse_v
            epoch_perc += perc_v
            epoch_tot  += tot_v
            n_batches  += 1

            pbar.set_postfix({
                "mse":  f"{mse_v:.4f}",
                "perc": f"{perc_v:.4f}",
                "tot":  f"{tot_v:.4f}",
            })

        scheduler.step()

        avg_mse  = epoch_mse  / n_batches
        avg_perc = epoch_perc / n_batches
        avg_tot  = epoch_tot  / n_batches
        cur_lr   = scheduler.get_last_lr()[0]
        # What fraction of the total loss is perceptual?
        perc_frac = (LAMBDA_PERC * avg_perc) / (avg_tot + 1e-8) * 100

        # Checkpoint
        accelerator.save_state(str(CHECKPOINT_DIR))
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            save_path = MODEL_SAVE_DIR / f"swindit_perc_epoch_{epoch+1}.pt"
            torch.save(unwrapped.state_dict(), save_path)
            with open(resume_file, "w") as f:
                json.dump({"last_epoch": epoch}, f)

        accelerator.print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"mse={avg_mse:.5f} | "
            f"perc={avg_perc:.5f} | "
            f"total={avg_tot:.5f} | "
            f"perc_contrib={perc_frac:.1f}% | "
            f"lr={cur_lr:.2e} | "
            f"perc_batches={perc_steps}"
        )

        # Diagnostic: warn if perceptual loss is dominating or negligible
        if accelerator.is_main_process:
            if perc_frac > 60:
                log.warning(
                    f"  Perceptual loss is {perc_frac:.0f}% of total — "
                    "consider reducing LAMBDA_PERC to 0.02 in .env.local."
                )
            elif perc_frac < 5 and perc_steps > 0:
                log.warning(
                    f"  Perceptual loss is only {perc_frac:.0f}% of total — "
                    "consider raising LAMBDA_PERC to 0.1."
                )

    accelerator.print("Perceptual fine-tuning complete.")
    accelerator.print(
        "Pick the epoch with highest SSIM (run ddim_inference.py after each checkpoint)."
    )


if __name__ == "__main__":
    train()