import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from tqdm import tqdm
import os
import yaml
from types import SimpleNamespace

# Internal Imports
from src.datasets.coco_restoration import CocoAuthDataset
from src.models.bridge.auth_bridge import AuthBridge
from src.models.swin_dit.backbone import SwinDiT

def load_config(config_path: str) -> SimpleNamespace:
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d
    return dict_to_namespace(config_dict)

def run_overfit_training():
    # Optimization for RTX 50-series (Blackwell)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda")
    
    config = load_config("src/configs/swin_dit_config.yaml")
    
    # Dataset Paths
    IMG_DIR = r"D:\Structa\claritycore\ai-engine\data\train2017"
    ANN_FILE = r"D:\Structa\claritycore\ai-engine\data\annotations\captions_train2017.json"

    # 1. Load Dataset
    full_dataset = CocoAuthDataset(img_dir=IMG_DIR, ann_file=ANN_FILE)
    indices = list(range(min(500, len(full_dataset))))
    
    # FORCE BATCH SIZE 1: 512x512 generates 65k tokens which OOMs 12GB VRAM
    train_loader = DataLoader(
        Subset(full_dataset, indices), 
        batch_size=1, 
        shuffle=True
    )

    # 2. Init Models
    embed_dim = config.model.embed_dim # 768
    bridge = AuthBridge(input_dim=embed_dim, output_dim=embed_dim).to(device)
    model = SwinDiT(config).to(device) 
    
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(bridge.parameters()), 
        lr=config.training.learning_rate
    )
    scaler = GradScaler()

    print(f"🚀 [Structa Labs] Final Handshake. Training at 256px on {device}...")
    model.train()
    bridge.train()

    for epoch in range(300): 
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            optimizer.zero_grad()
            
            # --- SPATIAL SCALING FOR VRAM STABILITY ---
            # Downsample to 256x256 to reduce token count from 65k to 16k
            hr_raw = batch['hr'].to(device)
            lr_raw = batch['lr'].to(device)
            
            hr_latents = F.interpolate(hr_raw, size=(256, 256), mode='bilinear', align_corners=False)
            lr_latents = F.interpolate(lr_raw, size=(256, 256), mode='bilinear', align_corners=False)
            batch_size = hr_latents.shape[0]
            
            # DYNAMIC ZERO PADDING: Uses .shape to prevent "512 vs 256" errors
            zeros = torch.zeros(
                (batch_size, 1, hr_latents.shape[2], hr_latents.shape[3]), 
                device=device
            )
            
            lr_4ch = torch.cat([lr_latents, zeros], dim=1)
            hr_4ch = torch.cat([hr_latents, zeros], dim=1)

            # Conditioning
            clip_emb = torch.randn(batch_size, 768, device=device)
            t_raw = torch.randn(batch_size, 768, device=device) 

            # Use bfloat16 for Blackwell stability and speed
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # 3. AuthBridge: Returns [B, 768]
                auth_cond = bridge(x=clip_emb, cond=t_raw)
                
                # 4. SwinDiT: Forward with precomputed condition
                output = model(lr_4ch, precomputed_cond=auth_cond)
                
                # Reconstruction Loss
                loss = F.mse_loss(output, hr_4ch)

            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

            # FORCE MEMORY CLEANUP
            del lr_4ch, hr_4ch, output, loss, zeros
            torch.cuda.empty_cache()

        # Save Checkpoint
        if epoch % 10 == 0:
            exp_dir = r"D:\Structa\claritycore\ai-engine\experiments"
            os.makedirs(exp_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(), 
                'bridge_state_dict': bridge.state_dict()
            }, os.path.join(exp_dir, "auth_swindiff_final.pt"))

    print("🏁 Training Complete. Structa Labs model ready.")

if __name__ == "__main__":
  run_overfit_training()