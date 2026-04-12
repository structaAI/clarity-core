import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import SiglipModel, SiglipProcessor
from diffusers import AutoencoderKL

# --- LOCAL IMPORTS ---
from src.models.swin_dit import SwinDiT
from src.models.bridge import AuthBridge

# --- PATHS ---
SIGLIP_PATH = r"D:\Structa\claritycore\ai-engine\src\models\CLIP\saved_models\siglip-so400m-patch14-384"
VAE_PATH = r"D:\Structa\claritycore\ai-engine\src\models\vae\saved_models"
CHECKPOINT_PATH = r"checkpoints/auth_swindiff_e1.pt" # Change this as you progress
SAMPLE_LR_PATH = r"D:\Structa\claritycore\ai-engine\data\val2017_lr\sample.jpg"

@torch.no_grad()
def verify_restoration(prompt="a high quality restored image"):
    device = "cuda"
    
    # 1. Load Components
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device).eval()
    siglip = SiglipModel.from_pretrained(SIGLIP_PATH).to(device).eval()
    processor = SiglipProcessor.from_pretrained(SIGLIP_PATH)
    
    # Initialize your models (ensure config matches training)
    model = SwinDiT(config).to(device).eval()
    fusion_layer = torch.nn.Linear(1152 + 1024, config.model.embed_dim).to(device).eval()
    
    # Load Weights from Training Checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state'])
    # If fusion_layer was in the checkpoint:
    # fusion_layer.load_state_dict(checkpoint['fusion_state']) 
    
    # 2. Prepare LR Input
    tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    lr_img = tf(Image.open(SAMPLE_LR_PATH).convert("RGB")).unsqueeze(0).to(device)

    # 3. Generate Conditioning (Same as Training)
    text_inputs = processor(text=prompt, return_tensors="pt").to(device)
    text_emb = siglip.get_text_features(**text_inputs)
    lr_feat = swin_lr_encoder(lr_img).mean(dim=[2,3]) # Your Swin-LR spatial anchor
    full_cond = fusion_layer(torch.cat([text_emb, lr_feat], dim=-1))

    # 4. Iterative Sampling Loop (DDIM-style)
    # Start with pure latent noise [1, 4, 128, 128] for 1024px output
    latents = torch.randn((1, 4, 128, 128), device=device)
    num_steps = 50
    
    print(f"🎨 Restoring {SAMPLE_LR_PATH}...")
    for i in range(num_steps):
        t = torch.full((1,), num_steps - i, device=device).long()
        
        # Predict noise using your integrated SwinDiT
        noise_pred = model(latents, t=t, precomputed_cond=full_cond)
        
        # Simple step (replace with a proper Scheduler if available)
        latents = latents - (0.02 * noise_pred)

    # 5. Decode back to Image Space
    # Remember the VAE scaling factor!
    decoded = vae.decode(latents / 0.18215).sample
    
    # Post-process and Save
    decoded = (decoded.clamp(-1, 1) + 1) / 2
    decoded = decoded.cpu().permute(0, 2, 3, 1).squeeze().numpy()
    result_img = Image.fromarray((decoded * 255).astype(np.uint8))
    result_img.save("restoration_test_epoch1.png")
    print("✅ Verification complete. Saved to 'restoration_test_epoch1.png'")

if __name__ == "__main__":
    verify_restoration()