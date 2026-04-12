import os
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
from transformers import SiglipModel, SiglipProcessor
from diffusers import AutoencoderKL # type: ignore
# --- 1. CONFIGURATION ---
class SimpleConfig:
    def __init__(self):
        self.model = self
        self.patch_size = 4
        self.in_channels = 4
        self.embed_dim = 1024
        self.depths = [2, 2, 6, 2]
        self.num_heads = [4, 8, 16, 32]
        self.window_size = 8
        self.use_pswa_bridge = True
        self.bridge_type = "auth"

config = SimpleConfig()

# --- 2. SETUP ---
DEVICE = torch.device("cuda")
SIGLIP_PATH = r"D:\Structa\claritycore\ai-engine\src\models\CLIP\saved_models\siglip-so400m-patch14-384"
VAE_PATH = r"D:\Structa\claritycore\ai-engine\src\models\vae\saved_models"
HR_DIR = r"D:\Structa\claritycore\ai-engine\data\train2017"
LR_DIR = r"D:\Structa\claritycore\ai-engine\data\train2017_lr"

torch.set_float32_matmul_precision('high')

# --- 3. DATASET ---
class AuthDataset(Dataset):
    def __init__(self, hr_dir, lr_dir):
        self.hr_dir, self.lr_dir = hr_dir, lr_dir
        self.imgs = [f for f in os.listdir(lr_dir) if f.endswith('.jpg')]
        self.tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, idx):
        name = self.imgs[idx]
        hr = self.tf(Image.open(os.path.join(self.hr_dir, name)).convert("RGB"))
        lr = self.tf(Image.open(os.path.join(self.lr_dir, name)).convert("RGB"))
        return lr, hr, name

    def __len__(self): return len(self.imgs)

# --- 4. INITIALIZATION ---
vae = AutoencoderKL.from_pretrained(VAE_PATH).to(DEVICE).eval() # type: ignore
siglip = SiglipModel.from_pretrained(SIGLIP_PATH).to(DEVICE).eval() # type: ignore
processor = SiglipProcessor.from_pretrained(SIGLIP_PATH)

# Main Models
from src.models.swin_dit.backbone import SwinDiT # Use your updated SwinDiT with AuthBridge
model = SwinDiT(config).to(DEVICE).train()
fusion_layer = nn.Linear(1152 + 1024, config.model.embed_dim).to(DEVICE).train()

# Placeholder for your Swin-LR branch (Ensure this is defined in your environment)
# swin_lr_encoder = ... 

optimizer = torch.optim.AdamW(list(model.parameters()) + list(fusion_layer.parameters()), lr=5e-5)
scaler = GradScaler()

# --- 5. TRAINING LOOP ---
loader = DataLoader(AuthDataset(HR_DIR, LR_DIR), batch_size=1, shuffle=True, pin_memory=True)

for epoch in range(12):
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}")
    for i, (lr, hr, _) in pbar:
        lr, hr = lr.to(DEVICE), hr.to(DEVICE)
        optimizer.zero_grad()
        
        with autocast(dtype=torch.bfloat16):
            with torch.no_grad():
                # Fix: Accessing distribution via index or latent_dist safely
                vae_out = vae.encode(hr)
                latents = (vae_out.latent_dist.sample() if hasattr(vae_out, 'latent_dist') else vae_out[0].sample()) * 0.18215
                
                # Fix: SigLIP positional call for return_tensors
                text_in = processor(["high quality restoration"], padding="max_length", return_tensors="pt").to(DEVICE)
                text_emb = siglip.get_text_features(**text_in) # [1, 1152]
                
                # Spatial branch
                lr_feat = swin_lr_encoder(lr).mean(dim=[2,3]) # [1, 1024]
            
            # Fix: Cat logic (using a list)
            full_cond = fusion_layer(torch.cat([text_emb, lr_feat], dim=-1))
            
            t = torch.randint(0, 1000, (1,), device=DEVICE).long()
            noise = torch.randn_like(latents)
            noisy_latents = latents + noise
            
            pred = model(noisy_latents, t=t, precomputed_cond=full_cond)
            loss = nn.MSELoss()(pred, noise)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    torch.save(model.state_dict(), f"checkpoints/auth_e{epoch+1}.pt")