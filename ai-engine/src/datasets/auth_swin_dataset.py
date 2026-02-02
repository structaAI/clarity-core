import os
import torch
from torch.utils.data import Dataset

class AuthSwinDataset(Dataset):
  def __init__(self, hr_latent_dir, lr_latent_dir):
    """
    - hr_latent_dir: Path to HR .pt files on Drive
    - lr_latent_dir: Path to LR .pt files on Drive
    """
    self.hr_dir = hr_latent_dir
    self.lr_dir = lr_latent_dir
    
    # Match IDs based on HR folder
    self.ids = [os.path.splitext(f)[0] for f in os.listdir(self.hr_dir) if f.endswith('.pt')]

  def __len__(self):
    return len(self.ids)

  def __getitem__(self, idx):
    file_id = self.ids[idx]
    
    # Load the Target (High-Resolution Latent)
    hr_latent = torch.load(os.path.join(self.hr_dir, f"{file_id}.pt"), map_location='cpu')
    
    # Load the Condition (Degraded/Low-Resolution Latent)
    lr_latent = torch.load(os.path.join(self.lr_dir, f"{file_id}.pt"), map_location='cpu')

    # Squeeze to remove batch dimension if it was saved as (1, C, H, W)
    return hr_latent.squeeze(0), lr_latent.squeeze(0)