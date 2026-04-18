from pathlib import Path
from PIL import Image
from inference.integrated_inference import IntegratedInferencePipeline

CONFIG   = r"D:\Structa\claritycore\ai-engine\src\configs\swin_dit_config.yaml"
CKPT     = r"D:\Structa\claritycore\ai-engine\src\models\swin_dit\saved_models\auth_integrated_epoch_10.pt"
CLIP     = r"D:\Structa\claritycore\ai-engine\src\models\CLIP\saved_models\siglip-so400m-patch14-384"
VAE      = r"D:\Structa\claritycore\ai-engine\src\models\VAE\saved_models"   # same VAE used in cache_latents.py
LR_IMAGE = r"D:\Structa\claritycore\ai-engine\data\train2017_lr\000000000009.jpg"

pipeline = IntegratedInferencePipeline.from_checkpoint(
    config_path=CONFIG,
    checkpoint=CKPT,
    clip_path=CLIP,
    vae_path=VAE,
    device="cuda",
    dtype="bfloat16",
)

lr = Image.open(LR_IMAGE)

# Unconditional SR
out_uncond = pipeline.super_resolve_to_image(lr)
out_uncond.save("result_uncond.png")

# Text-guided SR
out_cond = pipeline.super_resolve_to_image(
    lr,
    prompt="sharp, detailed photograph with clear textures",
    guidance_scale=7.5,
)
out_cond.save("result_cond.png")

print("Done — check result_uncond.png and result_cond.png")
