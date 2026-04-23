"""
benchmark_checkpoints.py
=========================
Scans all .pt checkpoint files in MODEL_SAVE_DIR, runs identical DDPM
inference on each using the same latent sample, and prints a ranked
results table showing PSNR and SSIM for every checkpoint.

This lets you recover the best checkpoint when you've lost track of which
one performs best, without having to run inference scripts manually for each.

What it does
------------
1. Discovers all *.pt files in the saved_models directory automatically.
2. Filters out any that can't be loaded as 8-channel SwinDiT checkpoints.
3. For each checkpoint, runs NUM_SAMPLES DDPM steps with multi-sample
   averaging and frequency smoothing — exactly the same pipeline as
   enhanced_inference.py.
4. Saves each checkpoint's best output to inference_outputs/benchmark/
   named after the checkpoint.
5. Prints a ranked table (best PSNR first) at the end.
6. Writes results to inference_outputs/benchmark/results.csv.

Usage
-----
  python src/inference/swin_dit_inference/benchmark_checkpoints.py

Required .env.local keys:
  VAE_PATH, LATENT_CACHE_DIR

Optional:
  MODEL_SAVE_DIR        — overrides default saved_models path
  NUM_SAMPLES           — DDPM runs to average per checkpoint (default 2,
                          keep low to make the sweep fast)
  NUM_INFERENCE_STEPS   — DDPM steps (default 1000)
  BLUR_SIGMA            — frequency smoothing sigma (default 0.8)
  BENCHMARK_SAMPLE      — fix a specific .safetensors file for fair comparison;
                          if not set, picks one randomly and uses it for ALL
  HR_SIZE / LR_SIZE     — default 512 / 128
"""

from __future__ import annotations

import csv
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from dotenv import load_dotenv
from PIL import Image, ImageFilter
from safetensors.torch import load_file

# ── Bootstrap ────────────────────────────────────────────────────
_here = Path(__file__).resolve()
_root = _here.parent
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────
VAE_PATH         = os.getenv("VAE_PATH", "")
LATENT_CACHE_DIR = os.getenv("LATENT_CACHE_DIR", "")
HR_SIZE          = int(os.getenv("HR_SIZE",  "512"))
LR_SIZE          = int(os.getenv("LR_SIZE",  "128"))
NUM_STEPS        = int(os.getenv("NUM_INFERENCE_STEPS", "1000"))
NUM_SAMPLES      = int(os.getenv("NUM_SAMPLES", "2"))   # keep low for speed
BLUR_SIGMA       = float(os.getenv("BLUR_SIGMA", "0.8"))
BENCHMARK_SAMPLE = os.getenv("BENCHMARK_SAMPLE", "")   # fix sample for fairness

_default_save_dir = (project_root / "src" / "models" / "swin_dit" / "saved_models")
MODEL_SAVE_DIR   = Path(os.getenv("MODEL_SAVE_DIR", str(_default_save_dir)))
OUTPUT_DIR       = project_root / "inference_outputs" / "benchmark"

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE   = torch.bfloat16
HR_LAT  = HR_SIZE // 8
LR_LAT  = LR_SIZE // 8


# ── Helpers ──────────────────────────────────────────────────────

def _move_diffusion(diffusion: GaussianDiffusion) -> None:
    for attr in [
        "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        "sqrt_recip_alphas", "posterior_variance",
        "posterior_log_variance_clipped",
        "posterior_mean_coef1", "posterior_mean_coef2",
    ]:
        if hasattr(diffusion, attr):
            setattr(diffusion, attr, getattr(diffusion, attr).to(DEVICE))


def _load_pair(path: Path):
    data     = load_file(str(path), device="cpu")
    hr       = data["hr"].unsqueeze(0).to(DEVICE, DTYPE)
    lr_small = data["lr_small"].unsqueeze(0).to(DEVICE, torch.float32)
    lr_up    = F.interpolate(lr_small, size=(HR_LAT, HR_LAT),
                             mode="bicubic", align_corners=False).to(DTYPE)
    return hr, lr_up


def _decode(vae, latent: torch.Tensor, scale: float = 0.13025) -> np.ndarray:
    z       = latent.to(DEVICE, torch.float32)
    decoded = vae.decode(z / scale).sample
    arr     = ((decoded.squeeze(0).clamp(-1, 1) + 1.0) / 2.0)
    return arr.cpu().float().detach().numpy().transpose(1, 2, 0)


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = float(np.mean((pred - gt) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        m = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
        return float(m(pred.float(), gt.float()).item())
    except ImportError:
        return float("nan")


def _pick_sample(cache_dir: str, fixed: str = "") -> Path:
    if fixed and Path(fixed).exists():
        return Path(fixed)
    import random
    files = sorted(Path(cache_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {cache_dir}")
    return random.choice(files)


def _try_load_checkpoint(model: SwinDiT, path: Path) -> bool:
    """
    Attempt to load checkpoint into model. Returns True on success.
    Silently skips files that aren't valid SwinDiT checkpoints.
    """
    try:
        ckpt  = torch.load(str(path), map_location="cpu", weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)

        # Must be a dict of tensors
        if not isinstance(state, dict):
            return False

        # Check patch embedding channel count — must be 8-ch
        proj_key = "patch_embedding.projection.weight"
        if proj_key not in state:
            return False
        if state[proj_key].shape[1] != 8:
            log.warning(f"  Skipping {path.name}: "
                        f"{state[proj_key].shape[1]}-ch patch embedding (need 8-ch)")
            return False

        mk, uk = model.load_state_dict(state, strict=False)
        # Allow minor mismatches (e.g. projection layers from integrated training)
        # but fail if more than 10 keys are missing — likely a wrong model entirely
        if len(mk) > 10:
            log.warning(f"  Skipping {path.name}: {len(mk)} missing keys — wrong model")
            return False

        return True

    except Exception as e:
        log.warning(f"  Skipping {path.name}: {e}")
        return False


def _run_checkpoint(
    model:     SwinDiT,
    diffusion: GaussianDiffusion,
    vae,
    hr_gt:     torch.Tensor,
    lr_up:     torch.Tensor,
) -> tuple[float, float, np.ndarray]:
    """
    Run NUM_SAMPLES DDPM passes on the given model, average, smooth.
    Returns (psnr, ssim, averaged_smoothed_array).
    """
    accumulated = np.zeros((HR_SIZE, HR_SIZE, 3), dtype=np.float32)

    def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_in = torch.cat([x, lr_up], dim=1)
        out  = model(x_in, t, clip_embeddings=None)
        return out[:, :4]

    for i in range(NUM_SAMPLES):
        torch.manual_seed(42 + i * 1000)
        with torch.no_grad():
            hr_pred = diffusion.p_sample_loop(
                model_fn=model_fn,
                shape=(1, 4, HR_LAT, HR_LAT),
                device=DEVICE,
                x_lr=lr_up,
                num_inference_steps=NUM_STEPS,
            )
        accumulated += _decode(vae, hr_pred)

    averaged = accumulated / NUM_SAMPLES
    smoothed = np.array(
        Image.fromarray((averaged * 255).clip(0, 255).astype(np.uint8))
        .filter(ImageFilter.GaussianBlur(radius=BLUR_SIGMA))
    ).astype(np.float32) / 255.0

    # Metrics
    gt_arr  = _decode(vae, hr_gt.to(torch.float32))
    psnr    = _psnr(smoothed, gt_arr)

    to_t    = T.ToTensor()
    ssim    = _ssim(
        to_t(Image.fromarray((smoothed * 255).clip(0, 255).astype(np.uint8))).unsqueeze(0),
        to_t(Image.fromarray((gt_arr   * 255).clip(0, 255).astype(np.uint8))).unsqueeze(0),
    )

    return psnr, ssim, smoothed


# ── Main ─────────────────────────────────────────────────────────

def main() -> None:
    # Validate required paths
    missing = [k for k, v in [
        ("VAE_PATH",         VAE_PATH),
        ("LATENT_CACHE_DIR", LATENT_CACHE_DIR),
    ] if not v or not Path(v).exists()]
    if missing:
        for m in missing:
            print(f"ERROR: {m} not set or not found in .env.local")
        sys.exit(1)

    if not MODEL_SAVE_DIR.exists():
        print(f"ERROR: MODEL_SAVE_DIR not found: {MODEL_SAVE_DIR}")
        sys.exit(1)

    # Discover all .pt checkpoints
    checkpoints = sorted(MODEL_SAVE_DIR.glob("*.pt"))
    if not checkpoints:
        print(f"ERROR: No .pt files found in {MODEL_SAVE_DIR}")
        sys.exit(1)

    log.info(f"Found {len(checkpoints)} checkpoint(s) in {MODEL_SAVE_DIR}")
    for ckpt in checkpoints:
        size_mb = ckpt.stat().st_size / 1e6
        log.info(f"  {ckpt.name:<45} {size_mb:>6.0f} MB")

    # Load config and build shared objects (reused across all checkpoints)
    swin_config = load_config(
        str(project_root / "src" / "configs" / "swin_dit_config.yaml"))

    diffusion = GaussianDiffusion(
        num_timesteps=swin_config.diffusion.num_sampling_steps,
        schedule=swin_config.diffusion.noise_schedule,
    )
    _move_diffusion(diffusion)

    from diffusers import AutoencoderKL  # type: ignore
    log.info(f"Loading VAE from: {VAE_PATH}")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True)
    vae = vae.to(DEVICE, torch.float32).eval()  # type: ignore
    log.info("  VAE loaded.")

    # Fix one sample for a fair apples-to-apples comparison
    sample_path = _pick_sample(LATENT_CACHE_DIR, BENCHMARK_SAMPLE)
    log.info(f"Benchmark sample: {sample_path.name}")
    log.info("  (All checkpoints evaluated on this same sample)")
    hr_gt, lr_up = _load_pair(sample_path)

    # Bicubic baseline (computed once)
    lr_arr     = _decode(vae, lr_up.to(torch.float32))
    lr_128_pil = Image.fromarray(
        (lr_arr * 255).clip(0, 255).astype(np.uint8)
    ).resize((LR_SIZE, LR_SIZE), Image.Resampling.LANCZOS)
    lr_bic_arr = np.array(
        lr_128_pil.resize((HR_SIZE, HR_SIZE), Image.Resampling.BICUBIC)
    ).astype(np.float32) / 255.0
    hr_gt_arr  = _decode(vae, hr_gt.to(torch.float32))

    psnr_bic = _psnr(lr_bic_arr, hr_gt_arr)
    to_t     = T.ToTensor()
    ssim_bic = _ssim(
        to_t(Image.fromarray((lr_bic_arr * 255).clip(0, 255).astype(np.uint8))).unsqueeze(0),
        to_t(Image.fromarray((hr_gt_arr  * 255).clip(0, 255).astype(np.uint8))).unsqueeze(0),
    )

    log.info(f"Bicubic baseline: PSNR={psnr_bic:.2f} dB  SSIM={ssim_bic:.4f}")

    # Save ground truth and bicubic once
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray((hr_gt_arr  * 255).clip(0, 255).astype(np.uint8)).save(
        OUTPUT_DIR / "groundtruth.png")
    Image.fromarray((lr_bic_arr * 255).clip(0, 255).astype(np.uint8)).save(
        OUTPUT_DIR / "bicubic_baseline.png")
    lr_128_pil.resize((LR_SIZE, LR_SIZE)).save(OUTPUT_DIR / "lr_input.png")

    # ── Benchmark loop ───────────────────────────────────────────
    model   = SwinDiT(swin_config).to(DEVICE, DTYPE)
    results = []   # list of dicts

    for ckpt_path in checkpoints:
        log.info(f"\n{'='*60}")
        log.info(f"Evaluating: {ckpt_path.name}")

        model.eval()
        if not _try_load_checkpoint(model, ckpt_path):
            log.info(f"  SKIPPED (incompatible checkpoint)")
            continue

        t0 = time.perf_counter()
        try:
            psnr, ssim, smoothed = _run_checkpoint(
                model, diffusion, vae, hr_gt, lr_up)
        except Exception as e:
            log.error(f"  FAILED during inference: {e}")
            continue

        elapsed  = time.perf_counter() - t0
        dpsnr    = psnr - psnr_bic

        # Save output image named after checkpoint
        out_name = ckpt_path.stem + "_output.png"
        Image.fromarray(
            (smoothed * 255).clip(0, 255).astype(np.uint8)
        ).save(OUTPUT_DIR / out_name)

        results.append({
            "checkpoint": ckpt_path.name,
            "psnr":       psnr,
            "ssim":       ssim,
            "delta_psnr": dpsnr,
            "time_s":     elapsed,
            "output":     out_name,
        })

        log.info(f"  PSNR : {psnr:.4f} dB  ({dpsnr:+.4f} vs bicubic)")
        log.info(f"  SSIM : {ssim:.4f}")
        log.info(f"  Time : {elapsed:.1f}s")

    if not results:
        print("\nNo checkpoints could be evaluated.")
        sys.exit(1)

    # ── Sort by PSNR ─────────────────────────────────────────────
    results.sort(key=lambda r: r["psnr"], reverse=True)

    # ── Print ranked table ────────────────────────────────────────
    print()
    print("=" * 80)
    print("  CHECKPOINT BENCHMARK — RANKED BY PSNR")
    print(f"  Sample  : {sample_path.name}")
    print(f"  Bicubic : {psnr_bic:.2f} dB  SSIM {ssim_bic:.4f}  (baseline)")
    print(f"  Samples : {NUM_SAMPLES} DDPM runs averaged per checkpoint")
    print("=" * 80)
    print(f"  {'Rank':<5} {'Checkpoint':<42} {'PSNR':>8} {'SSIM':>8} "
          f"{'vs Bic':>8} {'Time':>7}")
    print(f"  {'-'*78}")

    best_name = results[0]["checkpoint"]
    for rank, r in enumerate(results, 1):
        marker   = " ★" if rank == 1 else "  "
        ssim_str = f"{r['ssim']:.4f}" if not math.isnan(r['ssim']) else "  N/A"
        print(f"  {rank:<5}"
              f" {r['checkpoint']:<42}"
              f" {r['psnr']:>6.2f}dB"
              f" {ssim_str:>8}"
              f" {r['delta_psnr']:>+7.2f}dB"
              f" {r['time_s']:>6.0f}s"
              f"{marker}")

    print("=" * 80)
    print(f"\n  Best checkpoint: {best_name}")
    print(f"  PSNR: {results[0]['psnr']:.4f} dB  "
          f"SSIM: {results[0]['ssim']:.4f}  "
          f"({results[0]['delta_psnr']:+.2f} dB vs bicubic)")
    print(f"\n  Output images saved to: {OUTPUT_DIR}")
    print(f"  Open groundtruth.png alongside each *_output.png to compare.")

    # ── Write CSV ─────────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "checkpoint", "psnr", "ssim", "delta_psnr", "time_s", "output"])
        writer.writeheader()
        for rank, r in enumerate(results, 1):
            writer.writerow({"rank": rank, **r})

    print(f"  Full results written to: {csv_path}")
    print()

    # ── Final recommendation ──────────────────────────────────────
    print("  Update your .env.local:")
    print(f"  PRETRAINED_CHECKPOINT={MODEL_SAVE_DIR / best_name}")


if __name__ == "__main__":
    main()