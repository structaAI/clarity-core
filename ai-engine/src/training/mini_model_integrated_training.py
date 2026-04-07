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
    device = torch.device("cuda")
    config = load_config("src/configs/swin_dit_config.yaml")
    
    # Dataset Paths
    IMG_DIR = r"D:\Structa\claritycore\ai-engine\data\train2017"
    ANN_FILE = r"D:\Structa\claritycore\ai-engine\data\annotations\captions_train2017.json"

    full_dataset = CocoAuthDataset(img_dir=IMG_DIR, ann_file=ANN_FILE)
    indices = list(range(min(500, len(full_dataset))))
    train_loader = DataLoader(Subset(full_dataset, indices), batch_size=config.training.batch_size, shuffle=True)

    # Init Models
    embed_dim = config.model.embed_dim # 768
    bridge = AuthBridge(input_dim=embed_dim, output_dim=embed_dim).to(device)
    model = SwinDiT(config).to(device) 
    
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(bridge.parameters()), lr=config.training.learning_rate)
    scaler = GradScaler()

    print(f"🚀 [Structa Labs] Final Handshake Established. Training on {device}...")
    model.train()
    bridge.train()

    for epoch in range(300): 
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            optimizer.zero_grad()
            
            # 1. Prepare 4-channel Inputs (Matching SwinDiT in_chans=4)
            hr_latents = batch['hr'].to(device)
            lr_latents = batch['lr'].to(device)
            batch_size = hr_latents.shape[0]
            zeros = torch.zeros((batch_size, 1, 512, 512), device=device)
            lr_4ch = torch.cat([lr_latents, zeros], dim=1)
            hr_4ch = torch.cat([hr_latents, zeros], dim=1)

            # 2. Prepare Conditionings
            clip_emb = torch.randn(batch_size, 768, device=device) # Mock SigLIP
            t_raw = torch.randint(0, 1000, (batch_size,), device=device).float()

            with autocast(device_type='cuda'):
                # 3. AuthBridge: Refine CLIP with Time
                # x = Concept (768), cond = Signal (1)
                auth_cond = bridge(x=clip_emb, cond=t_raw)
                
                # 4. SwinDiT: Forward with precomputed condition
                output = model(lr_4ch, precomputed_cond=auth_cond)
                loss = F.mse_loss(output, hr_4ch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        if epoch % 10 == 0:
            exp_dir = r"D:\Structa\claritycore\ai-engine\experiments"
            os.makedirs(exp_dir, exist_ok=True)
            torch.save({'model_state_dict': model.state_dict(), 'bridge_state_dict': bridge.state_dict()}, 
                       os.path.join(exp_dir, "auth_swindiff_demo.pt"))

if __name__ == "__main__":
  run_overfit_training()