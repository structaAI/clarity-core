"""
verify_cache.py
===============
Run this BEFORE starting training to confirm the new SR latent cache
is correct. Takes about 5 seconds. Checks:

  1. Files exist and are readable
  2. Both 'hr' and 'lr_small' keys are present
  3. Shapes are correct (hr=[4,64,64], lr_small=[4,16,16])
  4. HR/LR 4x spatial ratio is maintained
  5. Values are finite (no NaN or Inf from the VAE encoder)
  6. Dataset __getitem__ returns lr upsampled to HR size correctly
  7. DataLoader collation works (no shape mismatch across files)

Usage:
  python src/training/verify_cache.py

Set LATENT_CACHE_DIR in .env.local first.
"""

from __future__ import annotations

import os
import sys
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from safetensors.torch import load_file

script_path  = Path(__file__).resolve()
project_root = script_path.parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv(dotenv_path=project_root / ".env.local")

LATENT_CACHE_DIR = os.getenv("LATENT_CACHE_DIR", "D:/Structa/claritycore/ai-engine/data/latents")
EXPECTED_HR_SPATIAL = int(os.getenv("HR_SIZE", "512")) // 8   # 64
EXPECTED_LR_SPATIAL = int(os.getenv("LR_SIZE", "128")) // 8   # 16
N_SPOT_CHECK = 20   # number of random files to inspect


def check(condition: bool, label: str, detail: str = "") -> bool:
    status = "\033[92m✔\033[0m" if condition else "\033[91m✘\033[0m"
    print(f"  {status}  {label}" + (f"  [{detail}]" if detail else ""))
    return condition


def main() -> None:
    print("\n\033[1mAuth-SwinDiff Cache Verification\033[0m")
    print(f"Cache : {LATENT_CACHE_DIR}")
    print(f"Expected HR spatial : {EXPECTED_HR_SPATIAL}×{EXPECTED_HR_SPATIAL}")
    print(f"Expected LR spatial : {EXPECTED_LR_SPATIAL}×{EXPECTED_LR_SPATIAL}")
    print("-" * 54)

    all_ok = True

    # ── Check 1: directory and files ──────────────────────────
    print("\n[1] Directory and file count")
    cache_dir = Path(LATENT_CACHE_DIR)
    all_ok &= check(cache_dir.exists(), "Cache directory exists", str(cache_dir))

    files = sorted(cache_dir.glob("*.safetensors"))
    all_ok &= check(len(files) > 0, f"Files found: {len(files)}")

    # ── Check 2: spot-check random files ──────────────────────
    print(f"\n[2] Spot-checking {N_SPOT_CHECK} random files")
    sample = random.sample(files, min(N_SPOT_CHECK, len(files)))
    shape_errors = []
    key_errors   = []
    value_errors = []

    for f in sample:
        try:
            data = load_file(str(f), device="cpu")
        except Exception as e:
            all_ok &= check(False, f"Readable: {f.name}", str(e))
            continue

        # Key presence
        has_hr       = "hr"       in data
        has_lr_small = "lr_small" in data
        has_old_lr   = "lr"       in data

        if not has_hr or not has_lr_small:
            key_errors.append(f"{f.name}: keys={list(data.keys())}")
            continue

        if has_old_lr:
            key_errors.append(f"{f.name}: has old 'lr' key alongside 'lr_small' — "
                              "old cache not fully replaced?")

        hr       = data["hr"]        # [4, H,   W  ]
        lr_small = data["lr_small"]  # [4, H/4, W/4]

        # Shape checks
        if hr.shape != (4, EXPECTED_HR_SPATIAL, EXPECTED_HR_SPATIAL):
            shape_errors.append(f"{f.name}: hr={tuple(hr.shape)}")
        if lr_small.shape != (4, EXPECTED_LR_SPATIAL, EXPECTED_LR_SPATIAL):
            shape_errors.append(f"{f.name}: lr_small={tuple(lr_small.shape)}")

        # 4× ratio check
        ratio_ok = (hr.shape[-1] // lr_small.shape[-1]) == 4
        if not ratio_ok:
            shape_errors.append(f"{f.name}: ratio={hr.shape[-1]}/{lr_small.shape[-1]} != 4")

        # Value finiteness
        if not torch.isfinite(hr).all() or not torch.isfinite(lr_small).all():
            value_errors.append(f"{f.name}: NaN or Inf detected")

    all_ok &= check(len(key_errors)   == 0, "All files have 'hr' and 'lr_small' keys",
                    f"{len(key_errors)} error(s)")
    all_ok &= check(len(shape_errors) == 0, f"All shapes correct  hr=[4,{EXPECTED_HR_SPATIAL},{EXPECTED_HR_SPATIAL}]  lr_small=[4,{EXPECTED_LR_SPATIAL},{EXPECTED_LR_SPATIAL}]",
                    f"{len(shape_errors)} error(s)")
    all_ok &= check(len(value_errors) == 0, "All values finite (no NaN/Inf)",
                    f"{len(value_errors)} error(s)")

    for e in key_errors[:3]:   print(f"      {e}")
    for e in shape_errors[:3]: print(f"      {e}")
    for e in value_errors[:3]: print(f"      {e}")

    # ── Check 3: Dataset __getitem__ and upsample ─────────────
    print("\n[3] Dataset load and upsample")
    try:
        sys.path.insert(0, str(project_root))
        from src.datasets.auth_swin_dataset import AuthSwinDataset
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = AuthSwinDataset(LATENT_CACHE_DIR)
            item = ds[0]

        hr_out = item["hr"]
        lr_out = item["lr"]

        all_ok &= check(
            hr_out.shape == (4, EXPECTED_HR_SPATIAL, EXPECTED_HR_SPATIAL),
            f"Dataset hr shape correct",
            str(tuple(hr_out.shape))
        )
        all_ok &= check(
            lr_out.shape == (4, EXPECTED_HR_SPATIAL, EXPECTED_HR_SPATIAL),
            f"Dataset lr (upsampled) shape correct",
            str(tuple(lr_out.shape))
        )

        # Check for old-format warning
        old_format_warned = any("old-format" in str(w.message) for w in caught)
        all_ok &= check(
            not old_format_warned,
            "No old-format 'lr' key warning (new SR cache confirmed)"
        )

        # Verify upsample is NOT trivially identical to hr (different content expected)
        max_diff = (hr_out - lr_out).abs().max().item()
        all_ok &= check(
            max_diff > 0.01,
            f"LR upsampled ≠ HR (max diff={max_diff:.4f}, confirms different content)"
        )

    except Exception as e:
        all_ok &= check(False, "Dataset __getitem__", str(e))

    # ── Check 4: DataLoader batch collation ───────────────────
    print("\n[4] DataLoader batch collation (batch_size=4)")
    try:
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)
        batch  = next(iter(loader))

        hr_b = batch["hr"]
        lr_b = batch["lr"]

        all_ok &= check(
            hr_b.shape == (4, 4, EXPECTED_HR_SPATIAL, EXPECTED_HR_SPATIAL),
            "HR batch shape",
            str(tuple(hr_b.shape))
        )
        all_ok &= check(
            lr_b.shape == (4, 4, EXPECTED_HR_SPATIAL, EXPECTED_HR_SPATIAL),
            "LR batch shape (upsampled)",
            str(tuple(lr_b.shape))
        )

        # Check channel-cat is feasible
        x_cat = torch.cat([hr_b, lr_b], dim=1)
        all_ok &= check(
            x_cat.shape == (4, 8, EXPECTED_HR_SPATIAL, EXPECTED_HR_SPATIAL),
            "Channel-cat [hr, lr] → [B,8,H,W]",
            str(tuple(x_cat.shape))
        )

    except Exception as e:
        all_ok &= check(False, "DataLoader collation", str(e))

    # ── Summary ───────────────────────────────────────────────
    print()
    print("=" * 54)
    if all_ok:
        print("\033[92m  ALL CHECKS PASSED — safe to start training.\033[0m")
        print(f"\n  Cache summary:")
        print(f"    Total latent pairs : {len(files)}")
        print(f"    HR latent shape    : [4, {EXPECTED_HR_SPATIAL}, {EXPECTED_HR_SPATIAL}]")
        print(f"    LR latent shape    : [4, {EXPECTED_LR_SPATIAL}, {EXPECTED_LR_SPATIAL}]  (stored)")
        print(f"    LR after upsample  : [4, {EXPECTED_HR_SPATIAL}, {EXPECTED_HR_SPATIAL}]  (at train time)")
        print(f"    SR factor          : {EXPECTED_HR_SPATIAL // EXPECTED_LR_SPATIAL}× in latent space")
        print(f"\n  Next step:")
        print(f"    python src/training/train_swindit.py")
        print(f"    (set PRETRAINED_CHECKPOINT=swindit_epoch_29.pt for fine-tuning)")
    else:
        print("\033[91m  CHECKS FAILED — do not start training.\033[0m")
        print("  Fix the errors above, then re-run this script.")
    print("=" * 54)
    print()

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()