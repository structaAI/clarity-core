"""
train_perceptual.py
====================
Fine-tunes SwinDiT with a combined MSE + perceptual loss.

Why this matters
----------------
Pure epsilon-MSE trains the model to predict the MEAN of all plausible
HR images consistent with the LR input. The mean of many sharp textures
is a blurry texture. This is the fundamental ceiling of MSE-only training.

Adding a perceptual loss computed on the predicted x0 gives the model
a direct signal: "the reconstructed image should have sharp, perceptually
realistic features". This is how SR3, StableSR, and Real-ESRGAN achieve
visual sharpness.

Loss formulation
----------------
  pred_x0  = (x_t - sqrt(1 - alpha_t) * eps_pred) / sqrt(alpha_t)
  L_mse    = MSE(eps_pred, eps)                       ← standard DDPM
  L_perc   = LPIPS(pred_x0_decoded, x0_decoded)       ← perceptual
  L_total  = L_mse + lambda_perc * L_perc

lambda_perc=0.05 is conservative. The perceptual loss is only applied
at LOW NOISE timesteps (t < T/4 = 250) where pred_x0 is meaningful.
At high noise timesteps pred_x0 is too corrupted for perceptual loss
to provide a useful signal, and including it would destabilise training.

Expected improvement
--------------------
  SSIM:  +0.05 to +0.10 over MSE-only
  PSNR:  roughly flat or slightly lower (expected — perceptual ≠ distortion)
  Visual: noticeably sharper edges and more defined textures

Training cost
-------------
10 epochs from swindit_v2_epoch_8.pt ≈ 7 hours overnight.

Usage
-----
  python src/training/train_perceptual.py

Required .env.local additions:
  PRETRAINED_CHECKPOINT=.../swindit_v2_epoch_8.pt
  LAMBDA_PERC=0.05        (perceptual loss weight, default 0.05)
  PERC_T_MAX=250          (only apply perc loss at t < this, default 250)
  NUM_EPOCHS=10           (fine-tuning epochs, default 10)
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
LEARNING_RATE         = float(os.getenv("LEARNING_RATE", "1e-5"))  # lower for perc fine-tune
LAMBDA_PERC           = float(os.getenv("LAMBDA_PERC",  "0.05"))
PERC_T_MAX            = int(os.getenv("PERC_T_MAX",   "250"))  # only at low-noise steps
GRAD_CLIP             = float(os.getenv("GRAD_CLIP",   "1.0"))
MODEL_SAVE_DIR        = project_root / "src" / "models" / "swin_dit" / "saved_models"
CHECKPOINT_DIR        = project_root / "checkpoints" / "perceptual" / "last"


# ── Perceptual Loss (LPIPS via VGG) ───────────────────────────────

class PerceptualLoss(nn.Module):
    """
    VGG-based perceptual loss using relu2_2 and relu3_3 features.

    Uses torchvision's pre-trained VGG-16. The model is frozen — only
    used as a fixed feature extractor, not trained.

    Input latents must be decoded through the VAE before computing
    this loss, since VGG expects 3-channel pixel images.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        import torchvision.models as tvm
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        # Use relu2_2 (idx 9) and relu3_3 (idx 16)
        self.slice1 = nn.Sequential(*list(vgg.children())[:10]).to(device).eval()
        self.slice2 = nn.Sequential(*list(vgg.children())[10:17]).to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)

        # ImageNet normalisation (VGG expects this)
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target : [B, 3, H, W]  pixel images in [0, 1]
        Returns scalar perceptual loss.
        """
        # Normalise to ImageNet stats
        pred   = (pred   - self.mean) / self.std # type: ignore
        target = (target - self.mean) / self.std  # type: ignore

        # Extract features at two VGG depths
        p1, t1 = self.slice1(pred),  self.slice1(target)
        p2, t2 = self.slice2(p1),    self.slice2(t1)

        return F.mse_loss(p1, t1) + F.mse_loss(p2, t2)


# ── Checkpoint loader (same universal loader as train_swindit.py) ─

def load_checkpoint(model: nn.Module, path: str) -> nn.Module:
    log.info(f"Loading checkpoint: {path}")
    ckpt  = torch.load(path, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    mk, uk = model.load_state_dict(state, strict=False)
    if mk:  log.warning(f"  Missing   : {mk}")
    if uk:  log.warning(f"  Unexpected: {uk}")
    log.info(f"  Loaded {len(state)} tensors.")
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
    T_train = swin_config.diffusion.num_sampling_steps   # 1000

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
        accelerator.print("WARNING: no checkpoint set — training from scratch.")

    # VAE (frozen, float32 for stability)
    if not VAE_PATH or not Path(VAE_PATH).exists():
        raise FileNotFoundError(
            f"VAE_PATH not found: '{VAE_PATH}'. "
            "Set VAE_PATH in .env.local — needed to decode latents for perceptual loss."
        )
    from diffusers import AutoencoderKL  # type: ignore
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True)
    vae = vae.to(accelerator.device, torch.float32).eval()  # type: ignore
    for p in vae.parameters():
        p.requires_grad_(False)
    VAE_SCALE = 0.13025

    # Perceptual loss (VGG, frozen)
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

    # Optimizer — lower LR than base training because we're fine-tuning
    # with a perceptual signal that can destabilise if LR is too high
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LEARNING_RATE, weight_decay=1e-2)

    # Fresh cosine schedule — do NOT resume old scheduler state
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Prepare (only model, optimizer, loader, scheduler — VAE+VGG stay frozen)
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
        setattr(diffusion, attr,
                getattr(diffusion, attr).to(accelerator.device))

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
        f"\n{'='*60}\n"
        f"  Perceptual Fine-Tuning\n"
        f"  Device       : {accelerator.device}\n"
        f"  Epochs       : {start_epoch} → {NUM_EPOCHS}\n"
        f"  LR           : {LEARNING_RATE}\n"
        f"  lambda_perc  : {LAMBDA_PERC}\n"
        f"  Perc t < T/4 : t < {PERC_T_MAX} (low-noise steps only)\n"
        f"  Dataset      : {len(dataset)} pairs\n"
        f"{'='*60}"
    )

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()

        epoch_mse  = 0.0
        epoch_perc = 0.0
        epoch_tot  = 0.0
        perc_steps = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}",
            disable=not accelerator.is_local_main_process,
        )

        for batch in pbar:
            optimizer.zero_grad()

            hr = batch["hr"]   # [B, 4, H, W]  clean HR latent
            lr = batch["lr"]   # [B, 4, H, W]  upsampled LR latent
            B  = hr.shape[0]

            # ── Forward diffusion ──────────────────────────────────
            t     = torch.randint(0, T_train, (B,),
                                  device=hr.device, dtype=torch.long)
            noise = torch.randn_like(hr)
            x_t   = diffusion.q_sample(hr, t, noise=noise)

            x_in  = torch.cat([x_t, lr], dim=1)   # [B, 8, H, W]

            # ── Model forward ──────────────────────────────────────
            eps_pred = model(x_in, t)[:, :4]       # [B, 4, H, W]

            # ── MSE loss on epsilon ────────────────────────────────
            loss_mse = F.mse_loss(eps_pred, noise)

            # ── Perceptual loss on pred_x0 (low-noise steps only) ──
            # At high noise (t > PERC_T_MAX), pred_x0 is too corrupted
            # to give a meaningful perceptual signal. Computing it anyway
            # wastes memory and destabilises training.
            loss_perc = torch.tensor(0.0, device=hr.device, dtype=hr.dtype)
            low_noise_mask = t < PERC_T_MAX   # [B] boolean

            if low_noise_mask.any() and LAMBDA_PERC > 0:
                idx = low_noise_mask.nonzero(as_tuple=True)[0]

                # Reconstruct pred_x0 from eps_pred for selected items
                at   = diffusion._extract(
                    diffusion.sqrt_alphas_cumprod, t[idx], hr[idx].shape)
                s1mt = diffusion._extract(
                    diffusion.sqrt_one_minus_alphas_cumprod, t[idx], hr[idx].shape)

                pred_x0 = (x_t[idx] - s1mt * eps_pred[idx]) / at
                pred_x0 = pred_x0.clamp(-1.0, 1.0)

                # Decode through VAE (float32 for stability)
                with torch.no_grad():
                    def _decode_latent(z: torch.Tensor) -> torch.Tensor:
                        z_f32   = z.to(torch.float32)
                        decoded = vae.decode(z_f32 / VAE_SCALE).sample # type: ignore
                        # Map [-1,1] → [0,1] for VGG
                        return ((decoded.clamp(-1, 1) + 1.0) / 2.0).to(hr.dtype)

                    pred_pixel = _decode_latent(pred_x0)   # [B', 3, H*8, W*8]
                    gt_pixel   = _decode_latent(hr[idx])

                # Perceptual loss — VGG features (float32 for VGG stability)
                loss_perc = perc_loss_fn(
                    pred_pixel.float(), gt_pixel.float()
                ).to(hr.dtype)

                perc_steps += 1

            # ── Combined loss ──────────────────────────────────────
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

            pbar.set_postfix({
                "mse":  f"{mse_v:.4f}",
                "perc": f"{perc_v:.4f}",
                "tot":  f"{tot_v:.4f}",
            })

        scheduler.step()

        n       = len(train_loader)
        avg_mse = epoch_mse  / n
        avg_prc = epoch_perc / n
        avg_tot = epoch_tot  / n
        cur_lr  = scheduler.get_last_lr()[0]

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
            f"perc={avg_prc:.5f} | "
            f"total={avg_tot:.5f} | "
            f"lr={cur_lr:.2e} | "
            f"perc_batches={perc_steps}"
        )

    accelerator.print("Perceptual fine-tuning complete.")
    accelerator.print(
        f"Best checkpoint strategy: pick the epoch with highest SSIM "
        f"(not lowest total loss) — run ddim_inference.py after each epoch."
    )


if __name__ == "__main__":
    train()