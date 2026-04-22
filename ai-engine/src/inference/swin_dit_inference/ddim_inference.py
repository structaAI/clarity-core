"""
ddim_inference.py
==================
DDIM (Denoising Diffusion Implicit Models) super-resolution inference.

Why DDIM instead of DDPM:
  - DDPM needs 1000 steps for correct sampling (~8 seconds per image).
  - DDIM is a deterministic/semi-stochastic ODE solver that works on ANY
    DDPM-trained model with NO retraining.
  - Inference drops from 8s → ~1-2s per image at 200 steps.
  - Outputs are reproducible: same seed = same image every time.

Important: linear beta schedule and DDIM step count
  This model was trained with a linear beta schedule (beta_start=1e-4,
  beta_end=0.02). On this schedule, alphas_cumprod[980] ≈ 6e-5, meaning
  nearly all signal is gone by t=700. This schedule needs MORE DDIM steps
  than a cosine schedule (which Stable Diffusion uses).
  - 50 steps: insufficient, produces noisy output
  - 200 steps: recommended default
  - 500 steps: near-DDPM quality, 2× faster than DDPM-1000

DDIM update rule (Song et al. 2021, Eq. 12):
  pred_x0 = (x_t - sqrt(1-at) * eps_pred) / sqrt(at).clamp(1e-3)
  sigma_t  = eta * sqrt((1-at_m1)/(1-at)) * sqrt(1 - at/at_m1)
  x_{t-1}  = sqrt(at_m1)*pred_x0 + sqrt(1-at_m1-sigma_t^2)*eps_pred + sigma_t*noise

Key fix: initial x = randn(shape) — NOT LR-seeded.
  The LR image enters via 8-channel cat inside model_fn at every step.
  Mixing LR into the start with sqrt(alphas_cumprod[980]) ≈ 0.008
  gives it effectively zero weight while causing numerical instability.

Usage:
  python src/inference/swin_dit_inference/ddim_inference.py

Required .env.local keys:
  PRETRAINED_CHECKPOINT, VAE_PATH, LATENT_CACHE_DIR

Optional:
  NUM_INFERENCE_STEPS   — default 200 (use 500 for best quality)
  NUM_SAMPLES           — runs to average, default 4
  DDIM_ETA              — stochasticity: 0.2 recommended for linear schedule
  HR_SIZE / LR_SIZE     — default 512 / 128
"""

from __future__ import annotations

import logging
import math
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
from src.utils.config_manager_swin_dit import load_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────
CHECKPOINT       = os.getenv("PRETRAINED_CHECKPOINT", "")
VAE_PATH         = os.getenv("VAE_PATH", "")
LATENT_CACHE_DIR = os.getenv("LATENT_CACHE_DIR", "")
HR_SIZE          = int(os.getenv("HR_SIZE",  "512"))
LR_SIZE          = int(os.getenv("LR_SIZE",  "128"))
NUM_STEPS        = int(os.getenv("NUM_INFERENCE_STEPS", "200"))
NUM_SAMPLES      = int(os.getenv("NUM_SAMPLES", "4"))
ETA              = float(os.getenv("DDIM_ETA", "0.2"))   # mild stochasticity for linear schedule
BLUR_SIGMA       = float(os.getenv("BLUR_SIGMA", "0.8"))
OUTPUT_DIR       = project_root / "inference_outputs"
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE            = torch.bfloat16
HR_LAT           = HR_SIZE // 8   # 64
LR_LAT           = LR_SIZE // 8   # 16


# ── DDIM Scheduler ────────────────────────────────────────────────

class DDIMScheduler:
    """
    DDIM sampler built on top of the same linear beta schedule used
    during training. No model changes required.

    Parameters
    ----------
    num_train_steps : int   T used during training (1000)
    num_infer_steps : int   steps at inference (50 recommended)
    eta             : float stochasticity. 0 = deterministic DDIM,
                            1 = stochastic (≈ DDPM)
    """

    def __init__(
        self,
        num_train_steps: int = 1000,
        num_infer_steps: int = 50,
        eta: float = 0.0,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> None:
        self.T         = num_train_steps
        self.S         = num_infer_steps
        self.eta       = eta

        # Build the same linear schedule used during training
        betas              = torch.linspace(beta_start, beta_end, num_train_steps,
                                            dtype=torch.float32)
        alphas             = 1.0 - betas
        alphas_cumprod     = torch.cumprod(alphas, dim=0)

        self.alphas_cumprod = alphas_cumprod.to(DEVICE)

        # Choose S evenly-spaced timesteps from [0, T-1] in REVERSE order
        # e.g. T=1000, S=50 → step every 20: [980, 960, ..., 20, 0]
        step_ratio   = num_train_steps // num_infer_steps
        timesteps    = (torch.arange(0, num_infer_steps) * step_ratio).long()
        self.timesteps = timesteps.flip(0).to(DEVICE)   # high → low

    def _alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Cumulative alpha product at timestep t. Shape matches t."""
        return self.alphas_cumprod[t]

    @torch.no_grad()
    def step(
        self,
        eps_pred:  torch.Tensor,   # [B, 4, H, W]  model noise prediction
        t:         int,            # current scalar timestep
        x_t:       torch.Tensor,   # [B, 4, H, W]  current noisy latent
        t_prev:    int,            # next (lower) timestep, or -1 for final
    ) -> torch.Tensor:
        """Single DDIM reverse step — Song et al. 2021, Eq. 12.

        x_{t-1} = sqrt(at_m1) * pred_x0
                + sqrt(1 - at_m1 - sigma_t^2) * eps_pred   ← direction to x_t
                + sigma_t * noise                            ← stochasticity (eta)

        sigma_t = eta * sqrt((1-at_m1)/(1-at)) * sqrt(1 - at/at_m1)
        """
        at    = self._alpha(torch.tensor(t,      device=DEVICE)).float()
        at_m1 = self._alpha(torch.tensor(t_prev, device=DEVICE)).float() \
                if t_prev >= 0 else torch.ones(1, device=DEVICE)

        x_t_f32    = x_t.float()
        eps_f32    = eps_pred.float()

        # ── Predict clean x₀ ──────────────────────────────────────
        # Clamp at.sqrt() away from zero to prevent division explosion
        # at ~ 6e-5 at t=980 on the linear schedule — without clamping
        # pred_x0 = (x_t - 0.9999*eps) / 0.0077 amplifies noise by 130×
        sqrt_at     = at.sqrt().clamp(min=1e-3)
        sqrt_1mat   = (1 - at).sqrt()
        pred_x0     = (x_t_f32 - sqrt_1mat * eps_f32) / sqrt_at
        pred_x0     = pred_x0.clamp(-1.0, 1.0)   # hard clamp every step

        # ── Sigma for stochastic term ──────────────────────────────
        # sigma_t^2 = eta^2 * (1-at_m1)/(1-at) * (1 - at/at_m1)
        sigma_t = torch.zeros(1, device=DEVICE)
        if self.eta > 0 and t_prev >= 0:
            sigma_t = (self.eta
                       * ((1 - at_m1) / (1 - at).clamp(min=1e-8)).sqrt()
                       * (1 - at / at_m1.clamp(min=1e-8)).clamp(min=0).sqrt())

        # ── Direction pointing to x_t ──────────────────────────────
        dir_coeff = (1 - at_m1 - sigma_t**2).clamp(min=0).sqrt()
        dir_xt    = dir_coeff * eps_f32

        # ── Stochastic noise term ──────────────────────────────────
        noise = sigma_t * torch.randn_like(x_t_f32) if self.eta > 0 else 0.0

        x_prev = at_m1.sqrt() * pred_x0 + dir_xt + noise
        return x_prev.to(x_t.dtype)

    @torch.no_grad()
    def sample(
        self,
        model_fn,              # callable: (x_t, t_tensor) → eps [B,4,H,W]
        shape: tuple,          # (B, C, H, W)
        x_lr:  torch.Tensor,   # LR conditioning latent [B, 4, H, W] (injected via model_fn)
        seed:  int = 42,
    ) -> torch.Tensor:
        """Full DDIM reverse loop. Returns predicted x₀.

        Initialisation: pure Gaussian noise — the standard DDIM starting point.
        The LR image is NOT mixed into the initial noise. It enters the model
        at every step via 8-channel concatenation inside model_fn. Mixing LR
        into the start noise with alphas_cumprod[980] ≈ 6e-5 would give a
        weight of sqrt(6e-5) ≈ 0.008 — effectively zero. The previous seeding
        was mathematically correct but numerically harmful: the LR had no
        influence and the model still had to recover from near-pure noise in
        50 steps, causing error amplification.
        """
        torch.manual_seed(seed)
        B = shape[0]

        # ── Pure Gaussian noise start (standard DDIM) ──────────────
        x = torch.randn(shape, device=DEVICE, dtype=DTYPE)

        ts = self.timesteps.tolist()
        for i, t_val in enumerate(ts):
            t_prev = ts[i + 1] if i + 1 < len(ts) else -1
            t_tensor = torch.full((B,), t_val, device=DEVICE, dtype=torch.long)

            eps_pred = model_fn(x, t_tensor)   # [B, 4, H, W]
            x        = self.step(eps_pred, t_val, x, t_prev).to(DTYPE)

        return x


# ── Helpers ───────────────────────────────────────────────────────

def _load_pair(path: Path):
    data     = load_file(str(path), device="cpu")
    hr       = data["hr"].unsqueeze(0).to(DEVICE, DTYPE)
    lr_small = data["lr_small"].unsqueeze(0).to(DEVICE, torch.float32)
    lr_up    = F.interpolate(lr_small, size=(HR_LAT, HR_LAT),
                             mode="bicubic", align_corners=False).to(DTYPE)
    return hr, lr_up


def _decode(vae, latent: torch.Tensor, scale: float = 0.13025) -> np.ndarray:
    l_f32   = latent.to(DEVICE, torch.float32)
    decoded = vae.decode(l_f32 / scale).sample
    arr     = ((decoded.squeeze(0).clamp(-1, 1) + 1.0) / 2.0)
    return arr.cpu().float().detach().numpy().transpose(1, 2, 0)


def _psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    mse = np.mean((pred - gt) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)


def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        m = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
        return m(pred.float(), gt.float()).item()
    except ImportError:
        return float("nan")


def _pick_sample(cache_dir: str) -> Path:
    import random
    files = sorted(Path(cache_dir).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No .safetensors in {cache_dir}")
    return random.choice(files)


# ── Main ─────────────────────────────────────────────────────────

def main() -> None:
    missing = [k for k, v in [
        ("PRETRAINED_CHECKPOINT", CHECKPOINT),
        ("VAE_PATH",              VAE_PATH),
        ("LATENT_CACHE_DIR",      LATENT_CACHE_DIR),
    ] if not v or not Path(v).exists()]
    if missing:
        for m in missing:
            print(f"ERROR: {m} not set or not found.")
        sys.exit(1)

    log.info(f"Device      : {DEVICE}")
    log.info(f"DDIM steps  : {NUM_STEPS}  (eta={ETA})")
    log.info(f"Avg samples : {NUM_SAMPLES}")

    # Load config and model
    swin_config = load_config(
        str(project_root / "src" / "configs" / "swin_dit_config.yaml"))
    model = SwinDiT(swin_config).to(DEVICE, DTYPE).eval()
    ckpt  = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    mk, uk = model.load_state_dict(state, strict=False)
    if mk: log.warning(f"Missing   : {mk}")
    if uk: log.warning(f"Unexpected: {uk}")
    log.info("  SwinDiT loaded.")

    # Build DDIM scheduler
    ddim = DDIMScheduler(
        num_train_steps=swin_config.diffusion.num_sampling_steps,
        num_infer_steps=NUM_STEPS,
        eta=ETA,
    )
    log.info(f"  DDIM timesteps: {ddim.timesteps[:5].tolist()} ... "
             f"{ddim.timesteps[-3:].tolist()}")

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
    B, C, H, W   = hr_gt.shape

    # Model closure — LR conditioning baked in
    def model_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_in = torch.cat([x, lr_up], dim=1)   # [B, 8, H, W]
        out  = model(x_in, t, clip_embeddings=None)
        return out[:, :4]

    # Multi-sample averaging
    log.info(f"\nRunning {NUM_SAMPLES} DDIM-{NUM_STEPS} samples...")
    accumulated = np.zeros((HR_SIZE, HR_SIZE, 3), dtype=np.float32)
    import time
    t0 = time.perf_counter()

    for i in range(NUM_SAMPLES):
        seed = 42 + i * 1000
        log.info(f"  Sample {i+1}/{NUM_SAMPLES}  seed={seed}")
        hr_pred = ddim.sample(model_fn, (B, C, H, W), lr_up, seed=seed)
        accumulated += _decode(vae, hr_pred)

    elapsed = time.perf_counter() - t0
    averaged = accumulated / NUM_SAMPLES
    smoothed_pil = Image.fromarray(
        (averaged * 255).clip(0, 255).astype(np.uint8)
    ).filter(ImageFilter.GaussianBlur(radius=BLUR_SIGMA))

    log.info(f"  Total inference time : {elapsed:.1f}s  "
             f"({elapsed/NUM_SAMPLES:.1f}s per sample)")
    log.info(f"  vs DDPM-1000         : ~{NUM_SAMPLES * 8:.0f}s  "
             f"({NUM_SAMPLES * 8 / (elapsed + 1e-9):.1f}× faster)")

    # Baselines
    lr_decoded  = _decode(vae, lr_up.to(torch.float32))
    lr_128_pil  = Image.fromarray(
        (lr_decoded * 255).clip(0, 255).astype(np.uint8)
    ).resize((LR_SIZE, LR_SIZE), Image.Resampling.LANCZOS)
    lr_bic_pil  = lr_128_pil.resize((HR_SIZE, HR_SIZE), Image.Resampling.BICUBIC)
    hr_gt_pil   = Image.fromarray(
        (_decode(vae, hr_gt.to(torch.float32)) * 255).clip(0, 255).astype(np.uint8))

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "lr_128_input":        OUTPUT_DIR / "lr_128_input.png",
        "lr_bicubic_baseline": OUTPUT_DIR / "lr_bicubic_baseline.png",
        "hr_groundtruth":      OUTPUT_DIR / "hr_groundtruth.png",
        "hr_ddim_smoothed":    OUTPUT_DIR / "hr_ddim_smoothed.png",
    }
    lr_128_pil.save(paths["lr_128_input"])
    lr_bic_pil.save(paths["lr_bicubic_baseline"])
    hr_gt_pil.save(paths["hr_groundtruth"])
    smoothed_pil.save(paths["hr_ddim_smoothed"])

    for label, p in paths.items():
        log.info(f"  Saved {label:<24} → {p}")

    # Metrics
    to_t    = T.ToTensor()
    gt_t    = to_t(hr_gt_pil).unsqueeze(0)
    bic_t   = to_t(lr_bic_pil).unsqueeze(0)
    ddim_t  = to_t(smoothed_pil).unsqueeze(0)

    psnr_bic  = _psnr(bic_t.numpy()[0].transpose(1,2,0),
                      gt_t.numpy()[0].transpose(1,2,0))
    psnr_ddim = _psnr(ddim_t.numpy()[0].transpose(1,2,0),
                      gt_t.numpy()[0].transpose(1,2,0))
    ssim_bic  = _ssim(bic_t,  gt_t)
    ssim_ddim = _ssim(ddim_t, gt_t)

    print()
    print("=" * 62)
    print("  Auth-SwinDiff DDIM Inference Results")
    print("=" * 62)
    print(f"  Sample          : {sample_path.name}")
    print(f"  DDIM steps      : {NUM_STEPS}  (eta={ETA})")
    print(f"  Avg samples     : {NUM_SAMPLES}")
    print(f"  Inference time  : {elapsed:.1f}s total  "
          f"({elapsed/NUM_SAMPLES:.1f}s/sample)")
    print()
    print(f"  {'Method':<28} {'PSNR':>10} {'SSIM':>10} {'vs Bicubic':>12}")
    print(f"  {'-'*62}")
    rows = [
        ("Bicubic baseline",          psnr_bic,  ssim_bic,  0.0),
        (f"DDIM-{NUM_STEPS} ({NUM_SAMPLES}-avg+smoothed)",
                                      psnr_ddim, ssim_ddim, psnr_ddim - psnr_bic),
    ]
    for name, psnr, ssim, dp in rows:
        delta    = f"{dp:+.2f} dB" if dp != 0 else "baseline"
        ssim_str = f"{ssim:.4f}" if not math.isnan(ssim) else "  N/A"
        print(f"  {name:<28} {psnr:>8.2f}dB {ssim_str:>10} {delta:>12}")
    print()
    print("  Saved files:")
    for label, p in paths.items():
        print(f"    {label:<26}: {p.name}")
    print("=" * 62)


if __name__ == "__main__":
    main()