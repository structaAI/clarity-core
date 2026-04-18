"""
Inference test for IntegratedInferencePipeline.

Run from ai-engine/src/:
    python test_integrated.py

Optional env overrides (.env.local or shell):
    LR_IMAGE      Path to a real low-res image (PNG/JPG). If not set, a
                  synthetic latent is used so you can test without a VAE.
    VAE_PATH      Path to local SDXL-VAE directory. Required for pixel-
                  space input/output; latent-mode is used if not set.
    CKPT_EPOCH    Epoch number to load (default: 10).
    NUM_STEPS     Inference denoising steps (default: 50).
    GUIDANCE      CFG guidance scale (default: 7.5).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
from dotenv import load_dotenv

script_dir   = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

load_dotenv(dotenv_path=project_root / "ai-engine" / ".env.local")
load_dotenv(dotenv_path=project_root / ".env.local")

from inference.integrated_inference import IntegratedInferencePipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths — edit these or set via environment / .env.local
# ---------------------------------------------------------------------------

_BASE = script_dir

CONFIG    = str(_BASE / "configs" / "swin_dit_config.yaml")
EPOCH     = int(os.getenv("CKPT_EPOCH", "8"))
CKPT      = str(_BASE / "models" / "swin_dit" / "saved_models" / f"auth_integrated_epoch_{EPOCH}.pt")
CLIP_PATH = str(_BASE / "models" / "CLIP" / "saved_models" / "siglip-so400m-patch14-384")
VAE_PATH  = os.getenv("VAE_PATH", "")
LR_IMAGE  = os.getenv("LR_IMAGE", "")

NUM_STEPS = int(os.getenv("NUM_STEPS", "50"))
GUIDANCE  = float(os.getenv("GUIDANCE", "7.5"))

OUT_DIR = _BASE / "inference_outputs"
OUT_DIR.mkdir(exist_ok=True)

TEST_PROMPTS = [
    "sharp, high resolution photograph with clear textures and fine detail",
    "a crisp, detailed photograph, professionally restored",
]


def _check_paths() -> None:
    missing = []
    if not Path(CONFIG).exists():
        missing.append(f"Config:     {CONFIG}")
    if not Path(CKPT).exists():
        missing.append(f"Checkpoint: {CKPT}")
    if not Path(CLIP_PATH).exists():
        missing.append(f"CLIP model: {CLIP_PATH}")
    if missing:
        log.error("Missing required files:")
        for m in missing:
            log.error(f"  {m}")
        sys.exit(1)


def _load_pipeline() -> IntegratedInferencePipeline:
    vae = VAE_PATH if VAE_PATH and Path(VAE_PATH).exists() else None
    if not vae:
        log.warning("VAE_PATH not set or not found — running in latent mode.")

    pipeline = IntegratedInferencePipeline.from_checkpoint(
        config_path=CONFIG,
        checkpoint=CKPT,
        clip_path=CLIP_PATH,
        vae_path=vae,
        device="cuda",
        dtype="bfloat16",
    )
    log.info(f"Pipeline ready | device={pipeline.device} | steps={NUM_STEPS}")
    return pipeline


def _get_lr_input(pipeline: IntegratedInferencePipeline):
    if LR_IMAGE and Path(LR_IMAGE).exists() and pipeline.vae is not None:
        from PIL import Image
        log.info(f"Encoding LR image: {LR_IMAGE}")
        lr_latent = pipeline.encode_image(Image.open(LR_IMAGE).convert("RGB"))
        return lr_latent, True
    else:
        if LR_IMAGE and not Path(LR_IMAGE).exists():
            log.warning(f"LR_IMAGE not found: {LR_IMAGE} — using synthetic latent.")
        elif LR_IMAGE and pipeline.vae is None:
            log.warning("LR_IMAGE set but no VAE loaded — using synthetic latent.")
        log.info("Using synthetic random latent [1, 4, 64, 64].")
        latent = torch.randn(1, 4, 64, 64, device=pipeline.device, dtype=pipeline.dtype)
        return latent, True


def run_unconditional(pipeline: IntegratedInferencePipeline, lr_latent: torch.Tensor) -> None:
    log.info("--- Unconditional SR ---")
    hr_latent = pipeline.super_resolve(
        lr_latent,
        prompt=None,
        lr_is_latent=True,
        num_inference_steps=NUM_STEPS,
    )
    log.info(f"  Output latent: {tuple(hr_latent.shape)}")

    if pipeline.vae is not None:
        img = pipeline.decode_latent(hr_latent)
        out_path = OUT_DIR / f"epoch{EPOCH}_uncond_steps{NUM_STEPS}.png"
        img.save(str(out_path))
        log.info(f"  Saved: {out_path}")
    else:
        out_path = OUT_DIR / f"epoch{EPOCH}_uncond_steps{NUM_STEPS}_latent.pt"
        torch.save(hr_latent.cpu(), str(out_path))
        log.info(f"  Saved latent: {out_path}")


def run_conditional(pipeline: IntegratedInferencePipeline, lr_latent: torch.Tensor) -> None:
    for i, prompt in enumerate(TEST_PROMPTS):
        log.info(f"--- Conditional SR | guidance={GUIDANCE} ---")
        log.info(f"  Prompt: \"{prompt}\"")

        hr_latent = pipeline.super_resolve(
            lr_latent,
            prompt=prompt,
            guidance_scale=GUIDANCE,
            lr_is_latent=True,
            num_inference_steps=NUM_STEPS,
        )
        log.info(f"  Output latent: {tuple(hr_latent.shape)}")

        if pipeline.vae is not None:
            img = pipeline.decode_latent(hr_latent)
            out_path = OUT_DIR / f"epoch{EPOCH}_cond{i+1}_cfg{GUIDANCE}_steps{NUM_STEPS}.png"
            img.save(str(out_path))
            log.info(f"  Saved: {out_path}")
        else:
            out_path = OUT_DIR / f"epoch{EPOCH}_cond{i+1}_cfg{GUIDANCE}_steps{NUM_STEPS}_latent.pt"
            torch.save(hr_latent.cpu(), str(out_path))
            log.info(f"  Saved latent: {out_path}")


def main() -> None:
    log.info(f"Checkpoint epoch : {EPOCH}")
    log.info(f"Inference steps  : {NUM_STEPS}")
    log.info(f"Guidance scale   : {GUIDANCE}")
    log.info(f"Output dir       : {OUT_DIR}")

    _check_paths()
    pipeline = _load_pipeline()
    lr_latent, _ = _get_lr_input(pipeline)

    run_unconditional(pipeline, lr_latent)
    run_conditional(pipeline, lr_latent)

    log.info("Done. Check inference_outputs/")


if __name__ == "__main__":
    main()
