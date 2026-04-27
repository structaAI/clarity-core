"""
train_perceptual.py — Final Version
=====================================
Fine-tunes SwinDiT with MSE + VGG perceptual loss.

Loss formulation
----------------
  pred_x0  = (x_t - sqrt(1-at) * eps_pred) / sqrt(at)
  L_mse    = MSE(eps_pred, eps)
  L_perc   = VGG_feature_MSE(decode(pred_x0), decode(hr))   ← raw, no normalisation
  L_total  = L_mse + LAMBDA_PERC * L_perc

Calibration (from observed training values on this dataset)
-----------------------------------------------------------
  Raw VGG loss on this dataset     : ~45
  MSE at convergence               : ~0.140
  LAMBDA_PERC = 0.0008 gives       : ~20% perceptual contribution  ← default
  LAMBDA_PERC = 0.0005 gives       : ~14%  (use if MSE rises above 0.150)
  LAMBDA_PERC = 0.0013 gives       : ~30%  (use if perc_contrib < 10%)

  DO NOT add spatial normalisation (/ H*W) — it reduces signal by 4096×
  making the perceptual loss negligible regardless of lambda.

Perceptual loss only applied at t < PERC_T_MAX (250) where pred_x0 is
meaningful. At high noise steps pred_x0 is too corrupted.

Usage
-----
  python src/training/train_perceptual.py

Required .env.local
-------------------
  PRETRAINED_CHECKPOINT   path to swindit_v2_epoch_29.pt
  LATENT_CACHE_DIR        path to .safetensors latent pairs
  VAE_PATH                path to local AutoencoderKL directory

Optional .env.local (defaults shown)
-------------------------------------
  LEARNING_RATE=3e-5
  LAMBDA_PERC=0.0008
  PERC_T_MAX=250
  NUM_EPOCHS=10
  BATCH_SIZE=4
  GRAD_CLIP=1.0
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

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
NUM_EPOCHS            = int(os.getenv("NUM_EPOCHS",     "10"))
BATCH_SIZE            = int(os.getenv("BATCH_SIZE",     "4"))
LEARNING_RATE         = float(os.getenv("LEARNING_RATE", "3e-5"))
LAMBDA_PERC           = float(os.getenv("LAMBDA_PERC",  "0.0008"))  # calibrated: ~20% contrib
PERC_T_MAX            = int(os.getenv("PERC_T_MAX",    "250"))
GRAD_CLIP             = float(os.getenv("GRAD_CLIP",    "1.0"))
VAE_SCALE             = float(os.getenv("VAE_SCALE_FACTOR", "0.13025"))

MODEL_SAVE_DIR = project_root / "src" / "models" / "swin_dit" / "saved_models"
CHECKPOINT_DIR = project_root / "checkpoints" / "perceptual" / "last"


# ── Perceptual Loss ───────────────────────────────────────────────

class PerceptualLoss(nn.Module):
    """
    VGG-16 perceptual loss using relu2_2 and relu3_3 feature layers.

    Frozen — used only as a fixed feature extractor.
    Inputs must be decoded RGB pixel tensors in [0, 1] range.

    Raw output is ~45 on this dataset at 512px decode resolution.
    Use LAMBDA_PERC=0.0008 for ~20% contribution to total loss.
    DO NOT divide by spatial area — that kills the signal.
    """

    def __init__(self, device: torch.device) -> None:
        super().__init__()
        import torchvision.models as tvm
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        self.slice1 = nn.Sequential(*list(vgg.children())[:10]).to(device).eval()
        self.slice2 = nn.Sequential(*list(vgg.children())[10:17]).to(device).eval()
        for p in self.parameters():
            p.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std",  std)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target : [B, 3, H, W] in [0, 1].
        Returns scalar. Expected range ~45 on 512px images.
        """
        pred   = (pred   - self.mean) / self.std   # type: ignore[operator]
        target = (target - self.mean) / self.std   # type: ignore[operator]
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
            skipped.append(
                f"{k} shape mismatch: ckpt={list(v.shape)} "
                f"model={list(model_state[k].shape)}"
            )

    model.load_state_dict(model_state, strict=False)
    log.info(f"  Loaded {loaded}/{len(model_state)} tensors.")
    if skipped:
        for s in skipped[:5]:
            log.warning(f"  skip: {s}")
        if len(skipped) > 5:
            log.warning(f"  ... and {len(skipped) - 5} more.")
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

    # Validate required env vars up front — fail fast before any GPU work
    missing = []
    if not PRETRAINED_CHECKPOINT or not Path(PRETRAINED_CHECKPOINT).exists():
        missing.append(f"PRETRAINED_CHECKPOINT (got: '{PRETRAINED_CHECKPOINT}')")
    if not LATENT_CACHE_DIR:
        missing.append("LATENT_CACHE_DIR")
    if not VAE_PATH or not Path(VAE_PATH).exists():
        missing.append(f"VAE_PATH (got: '{VAE_PATH}')")
    if missing:
        raise EnvironmentError(
            "Missing or invalid required env vars:\n" +
            "\n".join(f"  {m}" for m in missing) +
            "\nSet these in .env.local before running."
        )

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
    model = load_checkpoint(model, PRETRAINED_CHECKPOINT)
    accelerator.print(f"Fine-tuning from: {Path(PRETRAINED_CHECKPOINT).name}")

    # VAE — frozen, float32
    from diffusers import AutoencoderKL  # type: ignore
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True)
    vae = vae.to(accelerator.device, torch.float32).eval()  # type: ignore
    for p in vae.parameters():
        p.requires_grad_(False)

    # Perceptual loss — frozen VGG
    perc_loss_fn = PerceptualLoss(device=accelerator.device)

    # Dataset
    dataset = AuthSwinDataset(LATENT_CACHE_DIR)
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=2,
        persistent_workers=True,
        multiprocessing_context="spawn" if os.name == "nt" else None,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

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
            start_epoch = 0

    accelerator.print(
        f"\n{'='*62}\n"
        f"  Perceptual Fine-Tuning\n"
        f"  Device        : {accelerator.device}\n"
        f"  Checkpoint    : {Path(PRETRAINED_CHECKPOINT).name}\n"
        f"  Epochs        : {start_epoch} → {NUM_EPOCHS}\n"
        f"  LR            : {LEARNING_RATE}\n"
        f"  LAMBDA_PERC   : {LAMBDA_PERC}  (target ~20% contrib to total loss)\n"
        f"  PERC_T_MAX    : t < {PERC_T_MAX}\n"
        f"  Dataset       : {len(dataset)} pairs\n"
        f"{'='*62}\n"
        f"  Expected after epoch 1:\n"
        f"    mse ~0.140 | perc ~45 | total ~0.176 | perc_contrib ~20%\n"
        f"  If perc_contrib < 10%: set LAMBDA_PERC=0.0013 and restart\n"
        f"  If perc_contrib > 50%: set LAMBDA_PERC=0.0005 and restart\n"
        f"  If mse rises above 0.150: set LAMBDA_PERC=0.0005 and restart\n"
        f"{'='*62}"
    )

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()

        epoch_mse   = 0.0
        epoch_perc  = 0.0
        epoch_tot   = 0.0
        perc_steps  = 0
        n_batches   = 0

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

            # Forward diffusion
            t     = torch.randint(0, T_train, (B,),
                                  device=hr.device, dtype=torch.long)
            noise = torch.randn_like(hr)
            x_t   = diffusion.q_sample(hr, t, noise=noise)

            # 8-channel input — backbone extracts lr_tokens internally
            x_in     = torch.cat([x_t, lr], dim=1)         # [B, 8, H, W]
            eps_pred = model(x_in, t)[:, :4]               # [B, 4, H, W]

            # MSE loss on epsilon
            loss_mse = F.mse_loss(eps_pred, noise)

            # Perceptual loss — low noise steps only
            loss_perc = torch.tensor(0.0, device=hr.device, dtype=hr.dtype)
            low_noise_mask = t < PERC_T_MAX

            if low_noise_mask.any() and LAMBDA_PERC > 0:
                idx = low_noise_mask.nonzero(as_tuple=True)[0]

                # Reconstruct pred_x0 from eps_pred
                at   = diffusion._extract(
                    diffusion.sqrt_alphas_cumprod,
                    t[idx], hr[idx].shape
                )
                s1mt = diffusion._extract(
                    diffusion.sqrt_one_minus_alphas_cumprod,
                    t[idx], hr[idx].shape
                )
                pred_x0 = (x_t[idx] - s1mt * eps_pred[idx]) / at.clamp(min=1e-3)
                pred_x0 = pred_x0.clamp(-1.0, 1.0)

                # Decode to pixel space for VGG — single float32 cast, no double cast
                with torch.no_grad():
                    def _decode(z: torch.Tensor) -> torch.Tensor:
                        decoded = vae.decode(z.float() / VAE_SCALE).sample  # type: ignore
                        return (decoded.clamp(-1, 1) + 1.0) / 2.0           # → [0, 1]

                    pred_pixel = _decode(pred_x0)
                    gt_pixel   = _decode(hr[idx])

                # Raw VGG loss — NO spatial normalisation
                # Expected ~45. Calibrated via LAMBDA_PERC=0.0008 → ~20% contrib
                loss_perc = perc_loss_fn(
                    pred_pixel.float(), gt_pixel.float()
                ).to(hr.dtype)

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
                "perc": f"{perc_v:.1f}",   # raw VGG value (~45), easy to sanity check
                "tot":  f"{tot_v:.4f}",
            })

        scheduler.step()

        avg_mse   = epoch_mse  / n_batches
        avg_perc  = epoch_perc / n_batches
        avg_tot   = epoch_tot  / n_batches
        cur_lr    = scheduler.get_last_lr()[0]
        perc_frac = (LAMBDA_PERC * avg_perc) / (avg_tot + 1e-8) * 100

        # Save checkpoint
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
            f"perc={avg_perc:.2f} | "       # raw VGG value — should be ~45
            f"total={avg_tot:.5f} | "
            f"perc_contrib={perc_frac:.1f}% | "
            f"lr={cur_lr:.2e} | "
            f"perc_batches={perc_steps}"
        )

        # Actionable diagnostics
        if accelerator.is_main_process:
            if perc_frac > 50:
                log.warning(
                    f"  perc_contrib={perc_frac:.0f}% is too high. "
                    f"Set LAMBDA_PERC=0.0005 in .env.local and restart."
                )
            elif perc_frac < 10 and perc_steps > 0:
                log.warning(
                    f"  perc_contrib={perc_frac:.0f}% is too low. "
                    f"Set LAMBDA_PERC=0.0013 in .env.local and restart."
                )
            elif avg_mse > 0.150:
                log.warning(
                    f"  mse={avg_mse:.5f} rising above threshold. "
                    f"Set LAMBDA_PERC=0.0005 in .env.local and restart."
                )
            else:
                log.info(
                    f"  perc_contrib={perc_frac:.1f}% — healthy. No changes needed."
                )

    accelerator.print("\nPerceptual fine-tuning complete.")
    accelerator.print(
        "Next: run ddim_inference.py on each swindit_perc_epoch_N.pt "
        "and pick the checkpoint with the highest SSIM."
    )


if __name__ == "__main__":
    train()