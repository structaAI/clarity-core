from torch.utils.data import Dataset
from pathlib import Path
from safetensors.torch import load_file

class AuthSwinDataset(Dataset):
  def __init__(self, latent_dir: str):
    self.latent_dir = Path(latent_dir)
    self.files = sorted([f for f in self.latent_dir.iterdir() if f.suffix == ".safetensors"])
    
    if not self.files:
      raise FileNotFoundError(f"No .safetensors found in {latent_dir}")

  def __len__(self) -> int:
    return len(self.files)

  def __getitem__(self, idx: int):
    file_path = self.files[idx]
    # zero-copy load for maximum throughput
    data = load_file(str(file_path), device="cpu")
    return {
      "hr": data["hr"], # Target
      "lr": data["lr"]  # Input
    }