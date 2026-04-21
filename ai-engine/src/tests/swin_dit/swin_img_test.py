import sys
import torch
from pathlib import Path
from PIL import Image

# 1. Setup Project Root for Imports
project_root = Path(r"D:\Structa\claritycore\ai-engine")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.inference.inference_pipeline import InferencePipeline

# 2. Initialize Pipeline
# Note: Ensure VAE_PATH in your .env or the yaml points to the folder with forward slashes
config_path = project_root / "src" / "configs" / "swin_dit_config.yaml"
pipeline = InferencePipeline.from_config(str(config_path))

# 3. Load Trained Weights
weights_path = project_root / "src" / "models" / "swin_dit" / "saved_models" / "swindit_epoch_30.pt"
print(f"🔄 Loading weights from: {weights_path}")

state = torch.load(str(weights_path), map_location=pipeline.device, weights_only=True)

# Handle Accelerate/Trainer wrapping
if "model_state_dict" in state:
    state = state["model_state_dict"]
elif "state_dict" in state:
    state = state["state_dict"]

# Load into the denoiser
pipeline.denoiser.load_state_dict(state)
pipeline.denoiser.eval() # Safety check
print("✅ Denoiser weights successfully injected.")

# 4. Run Super-Resolution
lr_path = project_root / "data" / "train2017_lr" / "000000000009.jpg"
if not lr_path.exists():
    raise FileNotFoundError(f"LR image not found at {lr_path}")

lr_img = Image.open(lr_path).convert("RGB")
print(f"🎨 Running 50-step inference on {lr_path.name}...")

# Use the pipeline to resolve
# Result will be a PIL Image because we use 'super_resolve_to_image'
result = pipeline.super_resolve_to_image(lr_img, num_inference_steps=50)

# 5. Save Output
output_path = Path("output_epoch30.png")
result.save(output_path)
print(f"🚀 Success! Image saved to {output_path.absolute()}")