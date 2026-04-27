"""
ddim_inference.py
==================
DDIM super-resolution inference for Auth-SwinDiff.

Two modes
---------
1. Single checkpoint (default)
   Set PRETRAINED_CHECKPOINT in .env.local and run normally.
   Evaluates one checkpoint on one random image and saves output images.

2. Epoch sweep (--sweep)
   Runs inference on every swindit_perc_epoch_N.pt found in saved_models,
   evaluates each on the same N fixed images (--num_eval_images, default 5),
   averages PSNR and SSIM across all images per epoch, prints a ranked table.

   python src/inference/swin_dit_inference/ddim_inference.py --sweep
   python src/inference/swin_dit_inference/ddim_inference.py --sweep --num_eval_images 10

   The same set of images is used for every epoch (random.seed=42) so the
   comparison is fair AND representative across multiple scenes.

Usage
-----
  # Single checkpoint (PRETRAINED_CHECKPOINT set in .env.local)
  python src/inference/swin_dit_inference/ddim_inference.py

  # Sweep all perceptual epochs, 5 images each
  python src/inference/swin_dit_inference/ddim_inference.py --sweep

  # Sweep with more images for a more reliable average
  python src/inference/swin_dit_inference/ddim_inference.py --sweep --num_eval_images 10

Required .env.local
-------------------
  PRETRAINED_CHECKPOINT   (single mode only)
  VAE_PATH
  LATENT_CACHE_DIR

Optional .env.local
-------------------
  NUM_INFERENCE_STEPS=200
  NUM_SAMPLES=4             number of DDIM runs to average per image
  DDIM_ETA=0.0              0.0=deterministic, 0.2=slight stochasticity
  BLUR_SIGMA=0.8
  HR_SIZE=512
  LR_SIZE=128
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

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

from src.models.swin_dit.backbone import SwinDiT
from src.models.diffusion.diffusion_engine import GaussianDiffusion
from src.utils.config_manager_swin_dit import load_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────
CHECKPOINT       = os.getenv("PRETRAINED_CHECKPOINT", "")
VAE_PATH         = os.getenv("VAE_PATH", "")
LATENT_CACHE_DIR = os.getenv("LATENT_CACHE_DIR", "")
HR_SIZE          = int(os.getenv("HR_SIZE",   "512"))
LR_SIZE          = int(os.getenv("LR_SIZE",   "128"))
NUM_STEPS        = int(os.getenv("NUM_INFERENCE_STEPS", "200"))
NUM_SAMPLES      = int(os.getenv("NUM_SAMPLES", "4"))
ETA              = float(os.getenv("DDIM_ETA", "0.0"))
BLUR_SIGMA       = float(os.getenv("BLUR_SIGMA", "0.8"))
OUTPUT_DIR       = project_root / "inference_outputs"
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE            = torch.bfloat16
HR_LAT           = HR_SIZE // 8
LR_LAT           = LR_SIZE // 8


# ── DDIM Scheduler ───────────────────────────────────────────────

class DDIMScheduler:
    """
    DDIM sampler that reads alphas_cumprod directly from the training
    GaussianDiffusion object — guaranteeing schedule consistency.
    """

    def __init__(
        self,
        diffusion: GaussianDiffusion,
        num_infer_steps: int = 200,
        eta: float = 0.0,
    ) -> None:
        self.eta = eta
        self.T   = diffusion.alphas_cumprod.shape[0]
        self.alphas_cumprod = diffusion.alphas_cumprod.float().to(DEVICE)

        step_ratio     = self.T // num_infer_steps
        timesteps      = (torch.arange(0, num_infer_steps) * step_ratio).long()
        self.timesteps = timesteps.flip(0).to(DEVICE)

        log.info(f"  DDIM: T={self.T}, S={num_infer_steps}, "
                 f"step_ratio={step_ratio}, eta={eta}")
        log.info(f"  Timesteps: {self.timesteps[:5].tolist()} ... "
                 f"{self.timesteps[-5:].tolist()}")
        log.info(f"  alphas_cumprod[first/last]: "
                 f"{self.alphas_cumprod[self.timesteps[0]]:.5f} / "
                 f"{self.alphas_cumprod[self.timesteps[-1]]:.5f}")

    def _at(self, t: int) -> torch.Tensor:
        return self.alphas_cumprod[t].float()

    @torch.no_grad()
    def step(
        self,
        eps_pred: torch.Tensor,
        t:        int,
        x_t:      torch.Tensor,
        t_prev:   int,
    ) -> torch.Tensor:
        at    = self._at(t)
        at_m1 = self._at(t_prev) if t_prev >= 0 else torch.ones(1, device=DEVICE)

        x_f32   = x_t.float()
        eps_f32 = eps_pred.float()

        sqrt_at   = at.sqrt().clamp(min=1e-4)
        sqrt_1mat = (1.0 - at).sqrt()
        pred_x0   = (x_f32 - sqrt_1mat * eps_f32) / sqrt_at
        pred_x0   = pred_x0.clamp(-1.0, 1.0)

        sigma_t = torch.zeros(1, device=DEVICE)
        if self.eta > 0 and t_prev >= 0:
            sigma_t = (self.eta
                       * ((1 - at_m1) / (1 - at).clamp(min=1e-8)).sqrt()
                       * (1 - at / at_m1.clamp(min=1e-8)).clamp(min=0.0).sqrt())

        dir_coeff = (1 - at_m1 - sigma_t ** 2).clamp(min=0.0).sqrt()
        noise     = torch.randn_like(x_f32) if self.eta > 0 else torch.zeros_like(x_f32)
        x_prev    = at_m1.sqrt() * pred_x0 + dir_coeff * eps_f32 + sigma_t * noise
        return x_prev.to(x_t.dtype)

    @torch.no_grad()
    def sample(
        self,
        model_fn,
        shape: tuple,
        seed: int = 42,
    ) -> torch.Tensor:
        torch.manual_seed(seed)
        x  = torch.randn(shape, device=DEVICE, dtype=DTYPE)
        ts = self.timesteps.tolist()
        for i, t_val in enumerate(ts):
            t_prev   = ts[i + 1] if i + 1 < len(ts) else -1
            t_tensor = torch.full((shape[0],), t_val, device=DEVICE, dtype=torch.long)
            eps_pred = model_fn(x, t_tensor)
            x        = self.step(eps_pred, t_val, x, t_prev)
        return x


# ── Helpers ──────────────────────────────────────────────────────

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


def _pick_samples(cache_dir: str, n: int) -> list[Path]:
    """Pick n files from cache_dir using a fixed seed — same set every call."""
    files = sorted(Path(cache_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors files in {cache_dir}")
    random.seed(42)
    return random.sample(files, min(n, len(files)))


def _load_model(checkpoint: str, swin_config) -> SwinDiT:
    model = SwinDiT(swin_config).to(DEVICE, DTYPE).eval()
    ckpt  = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    mk, uk = model.load_state_dict(state, strict=False)
    if mk: log.warning(f"  Missing   : {mk}")
    if uk: log.warning(f"  Unexpected: {uk}")
    return model


def _build_diffusion(swin_config) -> GaussianDiffusion:
    diffusion = GaussianDiffusion(
        num_timesteps=swin_config.diffusion.num_sampling_steps,
        schedule=swin_config.diffusion.noise_schedule,
    )
    for attr in [
        "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        "sqrt_recip_alphas", "posterior_variance",
        "posterior_log_variance_clipped",
        "posterior_mean_coef1", "posterior_mean_coef2",
    ]:
        if hasattr(diffusion, attr):
            setattr(diffusion, attr, getattr(diffusion, attr).to(DEVICE))
    return diffusion


def _run_one_image(
    model:       SwinDiT,
    ddim:        DDIMScheduler,
    vae,
    hr_gt:       torch.Tensor,
    lr_up:       torch.Tensor,
    save_dir:    Path,
    label:       str,
    save_images: bool = True,
) -> dict:
    """
    Run NUM_SAMPLES DDIM passes on one image, average, smooth, compute metrics.
    Returns dict with psnr_ddim, ssim_ddim, psnr_bic, ssim_bic.
    """
    B, C, H, W = hr_gt.shape

    def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_in = torch.cat([x, lr_up], dim=1)
        out  = model(x_in, t, clip_embeddings=None)
        return out[:, :4]

    accumulated = np.zeros((HR_SIZE, HR_SIZE, 3), dtype=np.float32)
    t0 = time.perf_counter()

    for i in range(NUM_SAMPLES):
        seed    = 42 + i * 1000
        hr_pred = ddim.sample(model_fn, (B, C, H, W), seed=seed)
        accumulated += _decode(vae, hr_pred)

    elapsed  = time.perf_counter() - t0
    averaged = accumulated / NUM_SAMPLES
    smoothed_pil = Image.fromarray(
        (averaged * 255).clip(0, 255).astype(np.uint8)
    ).filter(ImageFilter.GaussianBlur(radius=BLUR_SIGMA))

    # Baselines
    lr_arr     = _decode(vae, lr_up.to(torch.float32))
    lr_128_pil = Image.fromarray(
        (lr_arr * 255).clip(0, 255).astype(np.uint8)
    ).resize((LR_SIZE, LR_SIZE), Image.Resampling.LANCZOS)
    lr_bic_pil = lr_128_pil.resize((HR_SIZE, HR_SIZE), Image.Resampling.BICUBIC)
    hr_gt_pil  = Image.fromarray(
        (_decode(vae, hr_gt.to(torch.float32)) * 255).clip(0, 255).astype(np.uint8))

    if save_images:
        save_dir.mkdir(parents=True, exist_ok=True)
        smoothed_pil.save(save_dir / f"{label}_ddim_smoothed.png")
        if not (save_dir / "lr_128_input.png").exists():
            lr_128_pil.save(save_dir / "lr_128_input.png")
            lr_bic_pil.save(save_dir / "lr_bicubic_baseline.png")
            hr_gt_pil.save(save_dir / "hr_groundtruth.png")

    # Metrics
    to_t   = T.ToTensor()
    gt_t   = to_t(hr_gt_pil).unsqueeze(0)
    bic_t  = to_t(lr_bic_pil).unsqueeze(0)
    ddim_t = to_t(smoothed_pil).unsqueeze(0)

    psnr_bic  = _psnr(bic_t.numpy()[0].transpose(1, 2, 0),
                      gt_t.numpy()[0].transpose(1, 2, 0))
    psnr_ddim = _psnr(ddim_t.numpy()[0].transpose(1, 2, 0),
                      gt_t.numpy()[0].transpose(1, 2, 0))
    ssim_bic  = _ssim(bic_t,  gt_t)
    ssim_ddim = _ssim(ddim_t, gt_t)

    return {
        "label":      label,
        "psnr_ddim":  psnr_ddim,
        "ssim_ddim":  ssim_ddim,
        "psnr_bic":   psnr_bic,
        "ssim_bic":   ssim_bic,
        "elapsed":    elapsed,
    }


# ── Single mode ───────────────────────────────────────────────────

def run_single(args) -> None:
    missing = [k for k, v in [
        ("PRETRAINED_CHECKPOINT", CHECKPOINT),
        ("VAE_PATH",              VAE_PATH),
        ("LATENT_CACHE_DIR",      LATENT_CACHE_DIR),
    ] if not v or not Path(v).exists()]
    if missing:
        for m in missing:
            print(f"ERROR: {m} not set or not found.")
        sys.exit(1)

    log.info(f"Device: {DEVICE} | Steps: {NUM_STEPS} | Samples: {NUM_SAMPLES}")

    swin_config = load_config(
        str(project_root / "src" / "configs" / "swin_dit_config.yaml"))
    diffusion   = _build_diffusion(swin_config)
    ddim        = DDIMScheduler(diffusion, num_infer_steps=NUM_STEPS, eta=ETA)

    log.info(f"Loading SwinDiT: {CHECKPOINT}")
    model = _load_model(CHECKPOINT, swin_config)
    log.info("  SwinDiT loaded.")

    from diffusers import AutoencoderKL  # type: ignore
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True)
    vae = vae.to(DEVICE, torch.float32).eval()  # type: ignore
    log.info("  VAE loaded.")

    # Single mode: pick one random image (no fixed seed — truly random each run)
    files = sorted(Path(LATENT_CACHE_DIR).glob("*.safetensors"))
    if not files:
        print(f"ERROR: No .safetensors files in {LATENT_CACHE_DIR}")
        sys.exit(1)
    sample_path = random.choice(files)
    log.info(f"Sample: {sample_path.name}")

    hr_gt, lr_up = _load_pair(sample_path)
    label  = Path(CHECKPOINT).stem
    result = _run_one_image(
        model, ddim, vae, hr_gt, lr_up,
        save_dir=OUTPUT_DIR, label=label, save_images=True
    )

    print()
    print("=" * 64)
    print("  Auth-SwinDiff DDIM Inference Results")
    print("=" * 64)
    print(f"  Checkpoint  : {label}")
    print(f"  Sample      : {sample_path.name}")
    print(f"  DDIM steps  : {NUM_STEPS}  (eta={ETA})")
    print(f"  Avg samples : {NUM_SAMPLES}")
    print()
    print(f"  {'Method':<30} {'PSNR':>10} {'SSIM':>10} {'vs Bicubic':>12}")
    print(f"  {'-'*64}")
    for name, psnr, ssim, dp in [
        ("Bicubic baseline",
         result["psnr_bic"], result["ssim_bic"], 0.0),
        (f"DDIM-{NUM_STEPS} ({NUM_SAMPLES}-avg+smooth)",
         result["psnr_ddim"], result["ssim_ddim"],
         result["psnr_ddim"] - result["psnr_bic"]),
    ]:
        delta    = f"{dp:+.2f} dB" if dp else "baseline"
        ssim_str = f"{ssim:.4f}" if not math.isnan(ssim) else "N/A"
        print(f"  {name:<30} {psnr:>8.2f}dB {ssim_str:>10} {delta:>12}")
    print("=" * 64)
    print(f"\n  Output: {OUTPUT_DIR / f'{label}_ddim_smoothed.png'}")


# ── Sweep mode ────────────────────────────────────────────────────

def run_sweep(args) -> None:
    missing = [k for k, v in [
        ("VAE_PATH",         VAE_PATH),
        ("LATENT_CACHE_DIR", LATENT_CACHE_DIR),
    ] if not v or not Path(v).exists()]
    if missing:
        for m in missing:
            print(f"ERROR: {m} not set or not found.")
        sys.exit(1)

    # Find checkpoints
    sweep_dir   = Path(args.sweep_dir) if args.sweep_dir else (
        project_root / "src" / "models" / "swin_dit" / "saved_models"
    )
    prefix      = args.prefix
    checkpoints = sorted(
        sweep_dir.glob(f"{prefix}*.pt"),
        key=lambda p: int(p.stem.replace(prefix, ""))
    )
    if not checkpoints:
        print(f"ERROR: No checkpoints matching '{prefix}*.pt' in {sweep_dir}")
        sys.exit(1)

    n_images = args.num_eval_images
    log.info(f"Found {len(checkpoints)} checkpoints to sweep.")
    log.info(f"Evaluating each on {n_images} fixed images (seed=42).")

    swin_config = load_config(
        str(project_root / "src" / "configs" / "swin_dit_config.yaml"))
    diffusion   = _build_diffusion(swin_config)
    ddim        = DDIMScheduler(diffusion, num_infer_steps=NUM_STEPS, eta=ETA)

    from diffusers import AutoencoderKL  # type: ignore
    vae = AutoencoderKL.from_pretrained(VAE_PATH, local_files_only=True)
    vae = vae.to(DEVICE, torch.float32).eval()  # type: ignore
    log.info("  VAE loaded.")

    # Fixed image set — same for every epoch, different images each time
    eval_paths = _pick_samples(LATENT_CACHE_DIR, n_images)
    log.info(f"Eval images ({len(eval_paths)}):")
    for p in eval_paths:
        log.info(f"  {p.name}")

    sweep_output = OUTPUT_DIR / "sweep"
    results      = []

    for ckpt_path in checkpoints:
        label = ckpt_path.stem
        log.info(f"\n── {label} ──")
        model = _load_model(str(ckpt_path), swin_config)

        img_psnrs, img_ssims, img_psnrs_bic, img_ssims_bic = [], [], [], []

        for img_idx, sample_path in enumerate(eval_paths):
            hr_gt, lr_up = _load_pair(sample_path)

            # Save images only for first checkpoint to avoid disk spam
            save = (ckpt_path == checkpoints[0])
            img_label = f"{label}_img{img_idx+1}"

            r = _run_one_image(
                model, ddim, vae, hr_gt, lr_up,
                save_dir=sweep_output / label,
                label=img_label,
                save_images=save,
            )
            img_psnrs.append(r["psnr_ddim"])
            img_ssims.append(r["ssim_ddim"])
            img_psnrs_bic.append(r["psnr_bic"])
            img_ssims_bic.append(r["ssim_bic"])

            log.info(
                f"  img {img_idx+1}/{len(eval_paths)}: "
                f"PSNR={r['psnr_ddim']:.2f}dB  SSIM={r['ssim_ddim']:.4f}  "
                f"({r['elapsed']:.0f}s)"
            )

        avg_psnr     = float(np.mean(img_psnrs))
        avg_ssim     = float(np.mean(img_ssims))
        avg_psnr_bic = float(np.mean(img_psnrs_bic))
        avg_ssim_bic = float(np.mean(img_ssims_bic))

        log.info(
            f"  [{label}] avg PSNR={avg_psnr:.2f}dB  "
            f"avg SSIM={avg_ssim:.4f}  "
            f"(over {len(eval_paths)} images)"
        )

        results.append({
            "label":      label,
            "psnr_ddim":  avg_psnr,
            "ssim_ddim":  avg_ssim,
            "psnr_bic":   avg_psnr_bic,
            "ssim_bic":   avg_ssim_bic,
        })

        del model
        torch.cuda.empty_cache()

    # Rank by SSIM (primary), PSNR (tiebreak)
    ranked = sorted(
        results,
        key=lambda r: (r["ssim_ddim"], r["psnr_ddim"]),
        reverse=True
    )
    best = ranked[0]

    print()
    print("=" * 80)
    print("  Auth-SwinDiff Epoch Sweep Results")
    print("=" * 80)
    print(f"  Images evaluated per epoch : {len(eval_paths)}  (fixed set, seed=42)")
    print(f"  DDIM steps                 : {NUM_STEPS}  "
          f"(eta={ETA}, {NUM_SAMPLES} samples/image)")
    print()
    print(f"  {'Epoch':<32} {'PSNR (avg)':>12} {'SSIM (avg)':>12} "
          f"{'ΔPSNR vs bic':>14}")
    print(f"  {'-'*72}")

    for r in results:   # epoch order
        marker   = " ◀ BEST" if r["label"] == best["label"] else ""
        delta    = r["psnr_ddim"] - r["psnr_bic"]
        ssim_str = f"{r['ssim_ddim']:.4f}" if not math.isnan(r["ssim_ddim"]) else "N/A"
        print(f"  {r['label']:<32} {r['psnr_ddim']:>10.2f}dB "
              f"{ssim_str:>12} {delta:>+12.2f}dB{marker}")

    print()
    print(f"  Bicubic baseline (avg):  "
          f"PSNR={results[0]['psnr_bic']:.2f}dB  "
          f"SSIM={results[0]['ssim_bic']:.4f}")
    print()
    print(f"  ★  Best checkpoint : {best['label']}")
    print(f"     Avg PSNR = {best['psnr_ddim']:.2f} dB  "
          f"(+{best['psnr_ddim']-best['psnr_bic']:.2f} vs bicubic)")
    print(f"     Avg SSIM = {best['ssim_ddim']:.4f}  "
          f"(+{best['ssim_ddim']-best['ssim_bic']:.4f} vs bicubic)")
    print()
    print(f"  Set in .env.local:")
    print(f"    PRETRAINED_CHECKPOINT="
          f"{sweep_dir / (best['label'] + '.pt')}")
    print("=" * 80)
    print(f"\n  Output images saved to: {sweep_output}")


# ── Entry point ───────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Auth-SwinDiff DDIM Inference")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Sweep all perceptual epoch checkpoints and rank by SSIM"
    )
    parser.add_argument(
        "--num_eval_images", type=int, default=5,
        help="Number of images to evaluate per epoch in sweep mode (default: 5)"
    )
    parser.add_argument(
        "--sweep_dir", type=str, default="",
        help="Directory containing .pt checkpoints (default: auto-detected)"
    )
    parser.add_argument(
        "--prefix", type=str, default="swindit_perc_epoch_",
        help="Checkpoint filename prefix to match in sweep mode"
    )
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()