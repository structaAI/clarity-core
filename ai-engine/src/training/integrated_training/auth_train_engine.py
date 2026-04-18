from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------------
# 1. BOOTSTRAP (Must happen before 'src' imports)
# ---------------------------------------------------------------------------
script_path = Path(__file__).resolve()
# integrated_training is at src/training/integrated_training/ (3 levels from root)
project_root = script_path.parents[3]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env.local")

from src.datasets.auth_swin_dataset import AuthSwinDataset
from src.models.diffusion.diffusion_engine import GaussianDiffusion
from src.models.swin_dit.backbone import SwinDiT
from src.utils.config_manager_swin_dit import load_config

# ---------------------------------------------------------------------------
# 2. LOGGING & HYPER-PARAMETERS
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

NUM_EPOCHS:            int   = int(os.getenv("NUM_EPOCHS",    "15"))
LEARNING_RATE:         float = float(os.getenv("LEARNING_RATE",  "3e-5"))
BATCH_SIZE:            int   = int(os.getenv("BATCH_SIZE",    "4"))
NUM_WORKERS:           int   = int(os.getenv("NUM_WORKERS",   "2"))
CFG_DROPOUT:           float = float(os.getenv("CFG_DROPOUT",   "0.1"))
GRAD_CLIP:             float = float(os.getenv("GRAD_CLIP",     "1.0"))
GRAD_ACCUM:            int   = int(os.getenv("GRAD_ACCUM",    "1"))
PRETRAINED_CHECKPOINT: str   = os.getenv("PRETRAINED_CHECKPOINT", "")

DEFAULT_CLIP_PATH = str(project_root / "src" / "models" / "CLIP" / "saved_models" / "siglip-so400m-patch14-384")
CLIP_MODEL_PATH: str = os.getenv("CLIP_MODEL_SAVE_PATH", DEFAULT_CLIP_PATH)

LATENT_CACHE_DIR: str  = os.getenv("LATENT_CACHE_DIR", "")
CAPTIONS_JSON:    str  = os.getenv("CAPTIONS_JSON", "")
MODEL_SAVE_DIR:   Path = project_root / "src" / "models" / "swin_dit" / "saved_models"
CHECKPOINT_DIR:   Path = project_root / "checkpoints" / "auth_integrated" / "last"

# ---------------------------------------------------------------------------
# 3. TEXT ENCODER (Isolates Text Tower to avoid PixelValue errors)
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    def __init__(self, clip_path: str, model_dim: int) -> None:
        super().__init__()
        log.info(f"Loading text encoder from: {clip_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(clip_path, local_files_only=True)
        full_model = AutoModel.from_pretrained(clip_path, local_files_only=True)

        # Target the text sub-model directly to avoid vision encoder requirements
        if hasattr(full_model, "text_model"):
            self.backbone = full_model.text_model
        else:
            self.backbone = full_model

        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        with torch.no_grad():
            dummy = self.tokenizer(["probe"], return_tensors="pt", padding=True, truncation=True)
            output = self.backbone(**dummy)

            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                probe = output.pooler_output
            elif hasattr(output, "last_hidden_state"):
                probe = output.last_hidden_state.mean(1)
            else:
                probe = output[0] if isinstance(output, (list, tuple)) else output

        clip_dim = probe.shape[-1]
        log.info(f"  Text Model detected. Dim: {clip_dim} → SwinDiT dim: {model_dim}")

        self.projection = nn.Sequential(
            nn.Linear(clip_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def encode(self, prompts: List[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            output = self.backbone(**inputs)
            if hasattr(output, "pooler_output") and output.pooler_output is not None:
                raw = output.pooler_output
            else:
                raw = output.last_hidden_state.mean(1)
        return self.projection(raw.to(dtype))

# ---------------------------------------------------------------------------
# 4. TRAINING LOGIC
# ---------------------------------------------------------------------------

def train() -> None:
    if not LATENT_CACHE_DIR:
        raise ValueError("LATENT_CACHE_DIR is not set in your .env.local file.")

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=GRAD_ACCUM,
        project_dir=str(CHECKPOINT_DIR.parent),
    )

    if accelerator.is_main_process:
        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    config_path = project_root / "src" / "configs" / "swin_dit_config.yaml"
    swin_config = load_config(str(config_path))

    diffusion = GaussianDiffusion(
        num_timesteps=swin_config.diffusion.num_sampling_steps,
        schedule=swin_config.diffusion.noise_schedule,
    )

    model = SwinDiT(swin_config)

    if not Path(CLIP_MODEL_PATH).exists():
        raise FileNotFoundError(f"SigLIP weights missing at {CLIP_MODEL_PATH}")

    text_encoder = TextEncoder(clip_path=CLIP_MODEL_PATH, model_dim=swin_config.model.embed_dim)
    text_encoder.backbone.to(accelerator.device)

    dataset = AuthSwinDataset(
        LATENT_CACHE_DIR,
        captions_json=CAPTIONS_JSON if CAPTIONS_JSON else None,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        persistent_workers=NUM_WORKERS > 0,
    )

    trainable = list(model.parameters()) + list(text_encoder.projection.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    model, text_encoder.projection, optimizer, train_loader, scheduler = accelerator.prepare(
        model, text_encoder.projection, optimizer, train_loader, scheduler
    )

    # Sync diffusion buffers to the correct device
    for attr in ["betas", "alphas", "alphas_cumprod", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod"]:
        if hasattr(diffusion, attr):
            setattr(diffusion, attr, getattr(diffusion, attr).to(accelerator.device))

    # --- Resume logic ---
    start_epoch = 0
    resume_file = CHECKPOINT_DIR / "resume_metadata.json"
    if resume_file.exists():
        try:
            accelerator.load_state(str(CHECKPOINT_DIR))
            with open(resume_file, "r") as f:
                start_epoch = json.load(f).get("last_epoch", 0) + 1
            log.info(f"Resumed from checkpoint. Starting at Epoch {start_epoch + 1}")
        except Exception as e:
            log.warning(f"Resume failed: {e}. Starting from scratch.")

    caption_source = "COCO captions" if CAPTIONS_JSON else "fallback prompts (set CAPTIONS_JSON for real captions)"
    log.info(f"Starting Auth-SwinDiff Integrated Training — text source: {caption_source}")
    log.info(f"Dataset size: {len(dataset)} latent pairs | Steps/epoch: {len(train_loader)}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        model.train()
        text_encoder.projection.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", disable=not accelerator.is_local_main_process)

        for batch in pbar:
            with accelerator.accumulate(model):
                optimizer.zero_grad()

                hr, lr_lat = batch["hr"], batch["lr"]
                captions: List[str] = [
                    "" if torch.rand(1).item() < CFG_DROPOUT else c
                    for c in batch["caption"]
                ]

                t = torch.randint(0, swin_config.diffusion.num_sampling_steps, (hr.shape[0],), device=hr.device)

                noise = torch.randn_like(hr)
                x_t = diffusion.q_sample(hr, t, noise=noise)

                # [B, 8, H, W]: noisy HR + LR conditioning
                x_input = torch.cat([x_t, lr_lat], dim=1)

                clip_embed = text_encoder.encode(captions, accelerator.device, hr.dtype)

                pred = model(x_input, t, clip_embeddings=clip_embed)

                # Supervise HR noise prediction (channels 0-3 only)
                loss = F.mse_loss(pred[:, :4, ...], noise)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, GRAD_CLIP)

                optimizer.step()
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        scheduler.step()

        # --- Checkpointing ---
        accelerator.save_state(str(CHECKPOINT_DIR))

        if accelerator.is_main_process:
            with open(resume_file, "w") as f:
                json.dump({"last_epoch": epoch}, f)

            save_path = MODEL_SAVE_DIR / f"auth_integrated_epoch_{epoch + 1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                "projection_state_dict": accelerator.unwrap_model(text_encoder.projection).state_dict(),
            }, save_path)
            log.info(f"Epoch {epoch + 1}/{NUM_EPOCHS} | LR: {scheduler.get_last_lr()[0]:.2e} | Saved → {save_path.name}")

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    train()
