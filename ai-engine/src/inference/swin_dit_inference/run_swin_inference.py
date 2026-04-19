"""
Auth-SwinDiff Inference Evaluation — True SR Edition
=====================================================
Evaluates a trained SwinDiT checkpoint on genuine 4× super-resolution:
  128×128 LR image → 512×512 HR image

What this does:
  1. Loads checkpoint and VAE.
  2. Runs sanity checks (shape, NaN).
  3. Picks a random latent pair from the cache and runs SR.
  4. Decodes and saves:
       lr_input.png       — bicubic upsampled LR (what the eye sees as input)
       hr_groundtruth.png — the clean HR ground truth
       hr_output.png      — the model's SR prediction
  5. Prints PSNR/SSIM with correct interpretation for generative models.

Usage:
  python src/inference/swin_dit_inference/run_swin_inference.py

Required .env.local keys:
  PRETRAINED_CHECKPOINT   — path to swindit_epoch_N.pt
  VAE_PATH                — path to AutoencoderKL directory
  LATENT_CACHE_DIR        — path to .safetensors latent pairs

Optional:
  NUM_INFERENCE_STEPS     — default 1000 (must equal T for DDPM)
  HR_SIZE                 — default 512
  LR_SIZE                 — default 128
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from dotenv import load_dotenv
from PIL import Image
from safetensors.torch import load_file

# ---------------------------------------------------------------------------
# Bootstrap — adjust parents depth to match your file location
# ---------------------------------------------------------------------------
script_path  = Path(__file__).resolve()
project_root = script_path.parents[3]   # src/inference/swin_dit_inference/run_swin_inference.py

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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CHECKPOINT       = os.getenv("PRETRAINED_CHECKPOINT", "")
VAE_PATH         = os.getenv("VAE_PATH", "")
LATENT_CACHE_DIR = os.getenv("LATENT_CACHE_DIR", "")
HR_SIZE          = int(os.getenv("HR_SIZE",  "512"))
LR_SIZE          = int(os.getenv("LR_SIZE",  "128"))
# DDPM requires num_inference_steps == T (1000) for correct sampling.
# Using fewer steps (e.g. 200) underestimates posterior variance per step
# by ~stride× and produces near-random output. Do not lower this.
NUM_STEPS        = int(os.getenv("NUM_INFERENCE_STEPS", "1000"))
OUTPUT_DIR       = project_root / "inference_outputs"
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE            = torch.bfloat16

# Latent spatial sizes (VAE downscales by 8)
HR_LAT = HR_SIZE // 8   # 64
LR_LAT = LR_SIZE // 8   # 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _move_diffusion_buffers(diffusion: GaussianDiffusion, device: torch.device) -> None:
    """Move all GaussianDiffusion coefficient tensors to device."""
    for attr in [
        "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        "sqrt_recip_alphas", "posterior_variance",
        "posterior_log_variance_clipped",
        "posterior_mean_coef1", "posterior_mean_coef2",
    ]:
        setattr(diffusion, attr, getattr(diffusion, attr).to(device))


def _load_latent_pair(path: Path, device: torch.device, dtype: torch.dtype):
    """
    Load an HR/LR latent pair and return (hr, lr_up) where lr_up has been
    bilinear-upsampled to match hr's spatial dimensions.

    Supports both new format ('lr_small') and old format ('lr').
    """
    data = load_file(str(path), device="cpu")
    hr   = data["hr"].unsqueeze(0).to(device, dtype)   # [1, 4, 64, 64]

    if "lr_small" in data:
        # New format: true low-res latent [4, 16, 16]
        lr_small = data["lr_small"].unsqueeze(0).to(device, torch.float32)
        lr_up = F.interpolate(
            lr_small,
            size=(HR_LAT, HR_LAT),
            mode="bilinear",
            align_corners=False,
        ).to(dtype)   # [1, 4, 64, 64]
        mode = f"true SR  (LR latent {LR_LAT}×{LR_LAT} → upsample → {HR_LAT}×{HR_LAT})"
    elif "lr" in data:
        # Old format: LR already at HR spatial size — restoration only
        lr_up = data["lr"].unsqueeze(0).to(device, dtype)
        mode  = "restoration only  (LR and HR at same spatial size — not true SR)"
        log.warning("Old-format cache detected. Re-run cache_latents.py with LR_SIZE=128 for true SR.")
    else:
        raise KeyError(f"Neither 'lr_small' nor 'lr' found. Keys: {list(data.keys())}")

    return hr, lr_up, mode


def _decode_latent(vae, latent: torch.Tensor, scale: float = 0.13025) -> Image.Image:
    l_f32   = latent.to(DEVICE, torch.float32)
    decoded = vae.decode(l_f32 / scale).sample
    pixel   = decoded.squeeze(0).clamp(-1, 1)
    pixel   = ((pixel + 1.0) * 127.5).byte().cpu()
    return T.ToPILImage()(pixel)


def _psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 2.0) -> float:
    import math
    mse = F.mse_loss(pred.float(), target.float()).item()
    return float("inf") if mse == 0 else 10.0 * math.log10(max_val ** 2 / mse)


def _ssim(pred: torch.Tensor, target: torch.Tensor) -> float:
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


# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

def run_sanity_checks(model: SwinDiT, diffusion: GaussianDiffusion, swin_config) -> None:
    """
    Validate model forward pass shape and absence of NaN.

    Spatial size must satisfy: (H // patch_size) % window_size == 0
    Minimum: window_size × patch_size = 8 × 2 = 16
    """
    log.info("Running sanity checks...")
    model.eval()

    ws  = swin_config.model.window_size   # 8
    ps  = swin_config.model.patch_size    # 2
    H = W = ws * ps                       # 16 — smallest valid spatial size
    log.info(f"  Sanity spatial: {H}×{W}  (window={ws} × patch={ps})")

    dummy_hr = torch.randn(1, 4, H, W, device=DEVICE, dtype=DTYPE)
    dummy_lr = torch.randn(1, 4, H, W, device=DEVICE, dtype=DTYPE)
    dummy_t  = torch.tensor([500], device=DEVICE, dtype=torch.long)

    # Check 1: q_sample
    noise = torch.randn_like(dummy_hr)
    x_t   = diffusion.q_sample(dummy_hr, dummy_t, noise=noise)
    assert x_t.shape == dummy_hr.shape and not torch.isnan(x_t).any()
    log.info("  ✔ q_sample")

    # Check 2: denoiser forward
    with torch.no_grad():
        out = model(torch.cat([x_t, dummy_lr], dim=1), dummy_t)
    assert out.shape == (1, model.out_channels, H, W) and not torch.isnan(out).any()
    log.info(f"  ✔ Denoiser forward  {tuple(out.shape)}")

    # Check 3: single p_sample reverse step
    with torch.no_grad():
        x_prev = diffusion.p_sample(
            model_fn=lambda x, t: model(torch.cat([x, dummy_lr], dim=1), t)[:, :4],
            x_t=x_t[:, :4],
            t=dummy_t,
        )
    assert x_prev.shape == dummy_hr.shape and not torch.isnan(x_prev).any()
    log.info("  ✔ p_sample")
    log.info("All sanity checks passed.\n")


# ---------------------------------------------------------------------------
# Full SR inference on one sample
# ---------------------------------------------------------------------------

def run_inference(
    model:     SwinDiT,
    diffusion: GaussianDiffusion,
    vae,
    sample_path: Path,
    num_steps:   int,
) -> dict:
    log.info(f"Sample: {sample_path.name}")

    hr_gt, lr_up, mode = _load_latent_pair(sample_path, DEVICE, DTYPE)
    log.info(f"  Mode    : {mode}")
    log.info(f"  HR latent : {tuple(hr_gt.shape)}")
    log.info(f"  LR latent : {tuple(lr_up.shape)}  (after upsample)")

    B, C, H, W = hr_gt.shape

    model.eval()

    def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # lr_up is already at HR spatial size [1, 4, H, W]
        x_in = torch.cat([x, lr_up], dim=1)                    # [1, 8, H, W]
        out  = model(x_in, t, clip_embeddings=None)             # [1, 8, H, W]
        return out[:, :4]                                       # HR noise only

    log.info(f"Running reverse diffusion ({num_steps} steps)...")

    def progress(step, _x):
        if step % max(1, num_steps // 10) == 0:
            print(f"  {step/num_steps*100:5.1f}%  [{step}/{num_steps}]",
                  end="\r", flush=True)

    with torch.no_grad():
        hr_pred = diffusion.p_sample_loop(
            model_fn=model_fn,
            shape=(B, C, H, W),
            device=DEVICE,
            x_lr=lr_up,
            num_inference_steps=num_steps,
            progress_callback=progress,
        )
    print()

    # --- Latent metrics ---
    psnr_lat = _psnr(hr_pred, hr_gt)
    ssim_lat = _ssim(hr_pred, hr_gt)
    log.info(f"  Latent PSNR : {psnr_lat:.2f} dB")
    log.info(f"  Latent SSIM : {ssim_lat:.4f}")

    # --- Decode ---
    log.info("Decoding latents...")

    # For the LR visualisation: decode lr_up then downsample to show the
    # "true" 128px input upscaled bicubically — what a naive baseline would give.
    lr_decoded_512 = _decode_latent(vae, lr_up.to(torch.float32))
    lr_pil         = lr_decoded_512.resize((LR_SIZE, LR_SIZE), Image.Resampling.LANCZOS)  # 128×128
    lr_bicubic_pil = lr_pil.resize((HR_SIZE, HR_SIZE), Image.Resampling.BICUBIC)          # 512×512 bicubic baseline

    hr_gt_pil      = _decode_latent(vae, hr_gt.to(torch.float32))
    hr_pred_pil    = _decode_latent(vae, hr_pred)

    # --- Save ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "lr_128":       OUTPUT_DIR / "lr_128_input.png",
        "lr_bicubic":   OUTPUT_DIR / "lr_bicubic_baseline.png",
        "hr_gt":        OUTPUT_DIR / "hr_groundtruth.png",
        "hr_pred":      OUTPUT_DIR / "hr_output.png",
    }
    lr_pil.save(paths["lr_128"])
    lr_bicubic_pil.save(paths["lr_bicubic"])
    hr_gt_pil.save(paths["hr_gt"])
    hr_pred_pil.save(paths["hr_pred"])

    for label, p in paths.items():
        log.info(f"  Saved {label:14} → {p}")

    # --- Pixel metrics ---
    to_t = T.ToTensor()
    gt_t   = to_t(hr_gt_pil).unsqueeze(0)
    pred_t = to_t(hr_pred_pil).unsqueeze(0)
    bic_t  = to_t(lr_bicubic_pil).unsqueeze(0)

    psnr_pred = _psnr(pred_t * 2 - 1, gt_t * 2 - 1)
    psnr_bic  = _psnr(bic_t * 2 - 1, gt_t * 2 - 1)
    ssim_pred = _ssim(pred_t, gt_t)
    ssim_bic  = _ssim(bic_t,  gt_t)

    return {
        "psnr_latent": psnr_lat,  "ssim_latent": ssim_lat,
        "psnr_pred":   psnr_pred, "ssim_pred":   ssim_pred,
        "psnr_bic":    psnr_bic,  "ssim_bic":    ssim_bic,
        "mode":        mode,
        **{k: str(v) for k, v in paths.items()},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    missing = [k for k, v in [
        ("PRETRAINED_CHECKPOINT", CHECKPOINT),
        ("VAE_PATH",              VAE_PATH),
        ("LATENT_CACHE_DIR",      LATENT_CACHE_DIR),
    ] if not v or not Path(v).exists()]
    if missing:
        for m in missing:
            print(f"ERROR: {m} not set or path not found in .env.local")
        sys.exit(1)

    log.info(f"Device : {DEVICE}")
    log.info(f"Dtype  : {DTYPE}")
    log.info(f"Steps  : {NUM_STEPS}")
    log.info(f"HR     : {HR_SIZE}px → latent {HR_LAT}×{HR_LAT}")
    log.info(f"LR     : {LR_SIZE}px → latent {LR_LAT}×{LR_LAT} → upsampled to {HR_LAT}×{HR_LAT}")

    # --- Load config ---
    config_path = project_root / "src" / "configs" / "swin_dit_config.yaml"
    swin_config = load_config(str(config_path))

    # --- Diffusion ---
    diffusion = GaussianDiffusion(
        num_timesteps=swin_config.diffusion.num_sampling_steps,
        schedule=swin_config.diffusion.noise_schedule,
    )

    # --- SwinDiT ---
    log.info(f"Loading SwinDiT from: {CHECKPOINT}")
    model = SwinDiT(swin_config).to(DEVICE, DTYPE)
    ckpt  = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    missing_k, unexpected_k = model.load_state_dict(state, strict=False)
    if missing_k:     log.warning(f"  Missing keys    : {missing_k}")
    if unexpected_k:  log.warning(f"  Unexpected keys : {unexpected_k}")
    log.info("  SwinDiT loaded.")

    # --- VAE ---
    from diffusers import AutoencoderKL # type: ignore[import]
    log.info(f"Loading VAE from: {VAE_PATH}")
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True)
    vae = vae.to(DEVICE, torch.float32).eval() # type: ignore
    log.info("  VAE loaded.")

    # --- Move diffusion buffers to device ---
    _move_diffusion_buffers(diffusion, DEVICE)

    # --- Sanity checks ---
    run_sanity_checks(model, diffusion, swin_config)

    # --- Inference ---
    sample_path = _pick_sample(LATENT_CACHE_DIR)
    m = run_inference(model, diffusion, vae, sample_path, NUM_STEPS)

    import math
    # --- Report ---
    print()
    print("=" * 64)
    print("  Auth-SwinDiff SR Inference Results")
    print("=" * 64)
    print(f"  Sample          : {Path(sample_path).name}")
    print(f"  Mode            : {m['mode'][:50]}")
    print(f"  Inference steps : {NUM_STEPS}")
    print()
    print(f"  {'Metric':<22} {'Model SR':>12} {'Bicubic':>12}")
    print(f"  {'-'*48}")
    print(f"  {'Pixel PSNR (dB)':<22} {m['psnr_pred']:>12.2f} {m['psnr_bic']:>12.2f}")
    if not math.isnan(m['ssim_pred']):
        print(f"  {'Pixel SSIM':<22} {m['ssim_pred']:>12.4f} {m['ssim_bic']:>12.4f}")
    print(f"  {'Latent PSNR (dB)':<22} {m['psnr_latent']:>12.2f} {'N/A':>12}")
    print()
    print("  NOTE: DDPM is stochastic — it generates a valid HR image,")
    print("  not the exact ground-truth. Compare model vs bicubic, not")
    print("  model vs ground-truth, for a meaningful quality signal.")
    print()
    print("  Saved files")
    print(f"    LR 128px input      : {m['lr_128']}")
    print(f"    LR bicubic 512px    : {m['lr_bicubic']}")
    print(f"    HR ground truth     : {m['hr_gt']}")
    print(f"    HR model prediction : {m['hr_pred']}")
    print("=" * 64)
    print()
    print("  Visual checklist:")
    print("    ✓ hr_output has sharper edges than lr_bicubic_baseline")
    print("    ✓ hr_output has similar scene content to hr_groundtruth")
    print("    ✓ hr_output has finer texture than the bicubic upscale")
    if not math.isnan(m['ssim_pred']) and not math.isnan(m['ssim_bic']):
        delta_ssim = m['ssim_pred'] - m['ssim_bic']
        delta_psnr = m['psnr_pred'] - m['psnr_bic']
        print()
        print(f"  Model vs Bicubic: PSNR {delta_psnr:+.2f} dB, "
              f"SSIM {delta_ssim:+.4f}")
        if delta_psnr > 0:
            print("  → Model beats bicubic baseline ✓")
        else:
            print("  → Model below bicubic. Re-run cache_latents with LR_SIZE=128")
            print("    and retrain to enable true SR (not just restoration).")


if __name__ == "__main__":
    main()