import json
import random
from pathlib import Path
from typing import Dict, List, Optional

from safetensors.torch import load_file
from torch.utils.data import Dataset

_FALLBACK_PROMPTS: List[str] = [
    "a high quality, sharp, detailed photograph",
    "restored high resolution image with clear textures",
    "crisp, noise-free 4K photograph",
    "visually sharp and detailed image, professional quality",
]


def _build_caption_map(captions_json: str) -> Dict[str, str]:
    """Build stem → caption mapping from a COCO-format captions JSON file."""
    with open(captions_json, "r") as f:
        data = json.load(f)

    # Map image_id → stem (filename without extension)
    id_to_stem: Dict[int, str] = {
        img["id"]: Path(img["file_name"]).stem
        for img in data["images"]
    }

    # One caption per stem — pick the first annotation encountered
    stem_to_caption: Dict[str, str] = {}
    for ann in data["annotations"]:
        stem = id_to_stem.get(ann["image_id"])
        if stem and stem not in stem_to_caption:
            stem_to_caption[stem] = ann["caption"]

    return stem_to_caption


class AuthSwinDataset(Dataset):
    def __init__(self, latent_dir: str, captions_json: Optional[str] = None) -> None:
        self.latent_dir = Path(latent_dir)
        self.files = sorted(
            [f for f in self.latent_dir.iterdir() if f.suffix == ".safetensors"]
        )

        if not self.files:
            raise FileNotFoundError(f"No .safetensors found in {latent_dir}")

        self.caption_map: Dict[str, str] = {}
        if captions_json:
            self.caption_map = _build_caption_map(captions_json)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        file_path = self.files[idx]
        data = load_file(str(file_path), device="cpu")

        # Latent files are named latent_{stem}.safetensors
        stem = file_path.stem[len("latent_"):]
        caption = self.caption_map.get(stem, random.choice(_FALLBACK_PROMPTS))

        return {
            "hr": data["hr"],
            "lr": data["lr"],
            "caption": caption,
        }
