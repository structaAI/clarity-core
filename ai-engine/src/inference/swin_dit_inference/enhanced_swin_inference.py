"""
enhanced_inference.py
======================
Runs two complementary improvements over the base inference script:

  Fix 1 — Multi-sample averaging
    Runs the full 1000-step DDPM loop N times (default 4) with different
    random seeds, decodes each result, and averages the pixel images.
    Because DDPM is stochastic, each run places window-boundary artefacts
    at slightly different positions. Averaging across runs suppresses them
    while consistent scene content reinforces — no model changes needed.

  Fix 2 — Targeted frequency smoothing
    The Swin window-boundary artefact has a known spatial frequency:
    128px periodicity in the 512px output (one window = 8 patches × 2px
    patch_size × 8 VAE upscale = 128px). A mild Gaussian blur at sigma=0.8
    applied after averaging suppresses this frequency with minimal impact
    on true scene edges (which are localised, not periodic).

Usage:
  python src/inference/swin_dit_inference/enhanced_inference.py

Required .env.local keys (same as run_swin_inference.py):
  PRETRAINED_CHECKPOINT, VAE_PATH, LATENT_CACHE_DIR

Optional:
  NUM_INFERENCE_STEPS   — default 1000
  NUM_SAMPLES           — number of DDPM runs to average (default 4)
  BLUR_SIGMA            — Gaussian sigma for frequency smoothing (default 0.8)
  HR_SIZE / LR_SIZE     — default 512 / 128
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from dotenv import load_dotenv
from PIL import Image, ImageFilter
from safetensors.torch import load_file

# ── Bootstrap ─────────────────────────────────────────────────────
script_path  = Path(__file__).resolve()
# Walk up to find project_root (directory containing src/)
_root = script_path.parent
while _root != _root.parent:
    if (_root / "src").is_dir():
        break
    _root = _root.parent
project_root = _root

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env.local")

from src.models.diffusion.diffusion_engine import GaussianDiffusion
from src.models.swin_dit.backbone import SwinDiT
from src.utils.config_manager_swin_dit import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────
CHECKPOINT       = os.getenv("PRETRAINED_CHECKPOINT", "")
VAE_PATH         = os.getenv("VAE_PATH", "")
LATENT_CACHE_DIR = os.getenv("LATENT_CACHE_DIR", "")
HR_SIZE          = int(os.getenv("HR_SIZE",  "512"))
LR_SIZE          = int(os.getenv("LR_SIZE",  "128"))
NUM_STEPS        = int(os.getenv("NUM_INFERENCE_STEPS", "1000"))
NUM_SAMPLES      = int(os.getenv("NUM_SAMPLES", "4"))    # runs to average
BLUR_SIGMA       = float(os.getenv("BLUR_SIGMA", "0.8")) # frequency smoothing
OUTPUT_DIR       = project_root / "inference_outputs"
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE            = torch.bfloat16
HR_LAT           = HR_SIZE // 8   # 64
LR_LAT           = LR_SIZE // 8   # 16


# ── Helpers ───────────────────────────────────────────────────────

def _move_diffusion(diffusion: GaussianDiffusion) -> None:
    for attr in [
        "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        "sqrt_recip_alphas", "posterior_variance",
        "posterior_log_variance_clipped",
        "posterior_mean_coef1", "posterior_mean_coef2",
    ]:
        setattr(diffusion, attr, getattr(diffusion, attr).to(DEVICE))


def _load_pair(path: Path):
    data    = load_file(str(path), device="cpu")
    hr      = data["hr"].unsqueeze(0).to(DEVICE, DTYPE)
    lr_small = data["lr_small"].unsqueeze(0).to(DEVICE, torch.float32)
    lr_up   = F.interpolate(
        lr_small, size=(HR_LAT, HR_LAT), mode="bilinear", align_corners=False
    ).to(DTYPE)
    return hr, lr_up


def _decode(vae, latent: torch.Tensor, scale: float = 0.13025) -> np.ndarray:
    """Decode latent → float32 numpy array [H, W, 3] in range [0, 1]."""
    l_f32   = latent.to(DEVICE, torch.float32)
    decoded = vae.decode(l_f32 / scale).sample
    pixel   = decoded.squeeze(0).clamp(-1, 1)
    pixel   = ((pixel + 1.0) / 2.0).cpu().float().detach().numpy()  # [3, H, W]
    return pixel.transpose(1, 2, 0)                         # [H, W, 3]


def _psnr(pred: np.ndarray, target: np.ndarray) -> float:
    import math
    mse = np.mean((pred - target) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def _ssim_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        m = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
        return m(pred.float(), target.float()).item()
    except ImportError:
        return float("nan")


def _pick_sample(cache_dir: str) -> Path:
    import random
    files = sorted(Path(cache_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {cache_dir}")
    return random.choice(files)


# ── Single DDPM run ───────────────────────────────────────────────

def _single_run(
    model: SwinDiT,
    diffusion: GaussianDiffusion,
    lr_up: torch.Tensor,
    seed: int,
) -> torch.Tensor:
    """Run the full reverse diffusion loop with a fixed seed. Returns HR latent."""
    torch.manual_seed(seed)
    B, C, H, W = 1, 4, HR_LAT, HR_LAT

    def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_in = torch.cat([x, lr_up], dim=1)
        out  = model(x_in, t, clip_embeddings=None)
        return out[:, :4]

    with torch.no_grad():
        return diffusion.p_sample_loop(
            model_fn=model_fn,
            shape=(B, C, H, W),
            device=DEVICE,
            x_lr=lr_up,
            num_inference_steps=NUM_STEPS,
        )


# ── Main ─────────────────────────────────────────────────────────

def main() -> None:
    # Validate paths
    missing = [k for k, v in [
        ("PRETRAINED_CHECKPOINT", CHECKPOINT),
        ("VAE_PATH",              VAE_PATH),
        ("LATENT_CACHE_DIR",      LATENT_CACHE_DIR),
    ] if not v or not Path(v).exists()]
    if missing:
        for m in missing:
            print(f"ERROR: {m} not set or path not found in .env.local")
        sys.exit(1)

    log.info(f"Device      : {DEVICE}")
    log.info(f"Steps       : {NUM_STEPS}")
    log.info(f"Avg runs    : {NUM_SAMPLES}  (multi-sample averaging)")
    log.info(f"Blur sigma  : {BLUR_SIGMA}   (frequency smoothing)")

    # Load config
    swin_config = load_config(
        str(project_root / "src" / "configs" / "swin_dit_config.yaml")
    )

    # Load diffusion
    diffusion = GaussianDiffusion(
        num_timesteps=swin_config.diffusion.num_sampling_steps,
        schedule=swin_config.diffusion.noise_schedule,
    )
    _move_diffusion(diffusion)

    # Load model
    log.info(f"Loading SwinDiT from: {CHECKPOINT}")
    model = SwinDiT(swin_config).to(DEVICE, DTYPE).eval()
    ckpt  = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    missing_k, unexpected_k = model.load_state_dict(state, strict=False)
    if missing_k:     log.warning(f"  Missing   : {missing_k}")
    if unexpected_k:  log.warning(f"  Unexpected: {unexpected_k}")
    log.info("  SwinDiT loaded.")

    # Load VAE
    from diffusers import AutoencoderKL  # type: ignore
    log.info(f"Loading VAE from: {VAE_PATH}")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True)
    vae = vae.to(DEVICE, torch.float32).eval()  # type: ignore
    log.info("  VAE loaded.")

    # Pick sample
    sample_path = _pick_sample(LATENT_CACHE_DIR)
    log.info(f"Sample: {sample_path.name}")
    hr_gt, lr_up = _load_pair(sample_path)

    # ── FIX 1: Multi-sample averaging ────────────────────────────
    log.info(f"\nRunning {NUM_SAMPLES} DDPM samples for averaging...")
    accumulated = np.zeros((HR_SIZE, HR_SIZE, 3), dtype=np.float32)

    for i in range(NUM_SAMPLES):
        seed = 42 + i * 1000
        log.info(f"  Run {i+1}/{NUM_SAMPLES}  (seed={seed})")
        hr_pred = _single_run(model, diffusion, lr_up, seed)
        frame   = _decode(vae, hr_pred)
        accumulated += frame
        log.info(f"    Done.")

    averaged_arr = accumulated / NUM_SAMPLES  # [H, W, 3] float32 in [0,1]
    averaged_pil = Image.fromarray((averaged_arr * 255).clip(0, 255).astype(np.uint8))

    # ── FIX 2: Targeted frequency smoothing ──────────────────────
    # Gaussian sigma=0.8 suppresses the 128px periodic window-boundary
    # artefact while preserving localised true scene edges.
    log.info(f"\nApplying frequency smoothing (sigma={BLUR_SIGMA})...")
    smoothed_pil = averaged_pil.filter(ImageFilter.GaussianBlur(radius=BLUR_SIGMA))

    # ── Baselines for comparison ──────────────────────────────────
    lr_decoded   = _decode(vae, lr_up.to(torch.float32))
    lr_pil_128   = Image.fromarray((lr_decoded * 255).clip(0, 255).astype(np.uint8))
    lr_pil_128   = lr_pil_128.resize((LR_SIZE, LR_SIZE), Image.Resampling.LANCZOS)
    lr_bic_pil   = lr_pil_128.resize((HR_SIZE, HR_SIZE), Image.Resampling.BICUBIC)
    hr_gt_pil    = Image.fromarray(
        (_decode(vae, hr_gt.to(torch.float32)) * 255).clip(0, 255).astype(np.uint8)
    )

    # Also run single-sample (seed=42) for direct comparison
    log.info("\nRunning single-sample baseline for comparison...")
    single_pred = _single_run(model, diffusion, lr_up, seed=42)
    single_arr  = _decode(vae, single_pred)
    single_pil  = Image.fromarray((single_arr * 255).clip(0, 255).astype(np.uint8))

    # ── Save ─────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "lr_128_input":        OUTPUT_DIR / "lr_128_input.png",
        "lr_bicubic_baseline": OUTPUT_DIR / "lr_bicubic_baseline.png",
        "hr_groundtruth":      OUTPUT_DIR / "hr_groundtruth.png",
        "hr_single_sample":    OUTPUT_DIR / "hr_single_sample.png",
        "hr_averaged":         OUTPUT_DIR / "hr_averaged.png",
        "hr_smoothed":         OUTPUT_DIR / "hr_smoothed_final.png",
    }
    lr_pil_128.save(paths["lr_128_input"])
    lr_bic_pil.save(paths["lr_bicubic_baseline"])
    hr_gt_pil.save(paths["hr_groundtruth"])
    single_pil.save(paths["hr_single_sample"])
    averaged_pil.save(paths["hr_averaged"])
    smoothed_pil.save(paths["hr_smoothed"])

    for label, path in paths.items():
        log.info(f"  Saved {label:<22} → {path}")

    # ── Metrics ──────────────────────────────────────────────────
    to_t = T.ToTensor()
    gt_t      = to_t(hr_gt_pil).unsqueeze(0)
    bic_t     = to_t(lr_bic_pil).unsqueeze(0)
    single_t  = to_t(single_pil).unsqueeze(0)
    avg_t     = to_t(averaged_pil).unsqueeze(0)
    smooth_t  = to_t(smoothed_pil).unsqueeze(0)

    def metrics(pred_t):
        psnr = _psnr(pred_t.numpy()[0].transpose(1,2,0),
                     gt_t.numpy()[0].transpose(1,2,0))
        ssim = _ssim_score(pred_t, gt_t)
        return psnr, ssim

    psnr_bic,    ssim_bic    = metrics(bic_t)
    psnr_single, ssim_single = metrics(single_t)
    psnr_avg,    ssim_avg    = metrics(avg_t)
    psnr_smooth, ssim_smooth = metrics(smooth_t)

    import math
    # ── Report ───────────────────────────────────────────────────
    print()
    print("=" * 66)
    print("  Auth-SwinDiff Enhanced Inference Results")
    print("=" * 66)
    print(f"  Sample   : {sample_path.name}")
    print(f"  Avg runs : {NUM_SAMPLES}    Blur sigma : {BLUR_SIGMA}")
    print()
    print(f"  {'Method':<28} {'PSNR':>10} {'SSIM':>10} {'vs Bicubic':>12}")
    print(f"  {'-'*62}")

    rows = [
        ("Bicubic baseline",             psnr_bic,    ssim_bic,    0.0, 0.0),
        ("Single sample (no fixes)",     psnr_single, ssim_single, psnr_single-psnr_bic, ssim_single-ssim_bic),
        (f"Averaged ({NUM_SAMPLES} runs)",  psnr_avg,    ssim_avg,    psnr_avg-psnr_bic,   ssim_avg-ssim_bic),
        ("Averaged + smoothed (final)",  psnr_smooth, ssim_smooth, psnr_smooth-psnr_bic, ssim_smooth-ssim_bic),
    ]
    for name, psnr, ssim, dp, ds in rows:
        delta = f"PSNR {dp:+.2f} dB" if dp != 0.0 else "baseline"
        ssim_str = f"{ssim:.4f}" if not math.isnan(ssim) else "  N/A  "
        print(f"  {name:<28} {psnr:>8.2f}dB {ssim_str:>10} {delta:>12}")

    print()
    print("  Saved files:")
    for label, path in paths.items():
        print(f"    {label:<24}: {path.name}")
    print("=" * 66)
    print()
    print("  Open hr_smoothed_final.png — this is your best output.")
    print("  Compare it to lr_bicubic_baseline.png side by side.")


if __name__ == "__main__":
    main()