import torch
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image
import os
import random
from typing import Any, Dict, List, Optional, Callable, Sequence

class CocoAuthDataset(Dataset):
    def __init__(self, img_dir: str, ann_file: str, transform: Optional[Callable] = None, degradation_pipeline: Optional[Callable] = None) -> None:
        self.coco: COCO = COCO(ann_file)
        self.ids: List[int] = list(self.coco.imgs.keys())
        self.img_dir: str = img_dir
        self.transform: Optional[Callable] = transform
        self.degrade: Optional[Callable] = degradation_pipeline

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Initialize fallback values to prevent UnboundLocalError on failure
        caption = "A photograph"
        try:
            img_id: int = self.ids[index]
            img_info: Any = self.coco.loadImgs(img_id)[0]
            path: str = img_info['file_name']

            # Load and Force Resize to 512x512
            raw_img = Image.open(os.path.join(self.img_dir, path)).convert('RGB')
            hr_img = raw_img.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Get Caption
            ann_ids: List[int] = self.coco.getAnnIds(imgIds=img_id)
            anns: Sequence[Any] = self.coco.loadAnns(ann_ids)
            if anns:
                selected_ann = random.choice(anns)
                caption = str(dict(selected_ann).get('caption', "A photograph"))

            # Apply degradation
            if self.degrade:
                lr_img, deg_id, severity = self.degrade(hr_img)
            else:
                # Fallback: Simple downsample/upsample
                lr_img = hr_img.resize((128, 128)).resize((512, 512), Image.Resampling.NEAREST)
                deg_id, severity = 0, 0.5

            # Tensor Conversion
            hr_tensor = torch.from_numpy(np.array(hr_img)).permute(2, 0, 1).float() / 255.0
            lr_tensor = torch.from_numpy(np.array(lr_img)).permute(2, 0, 1).float() / 255.0

            return {
                "hr": hr_tensor,
                "lr": lr_tensor,
                "caption": caption,
                "deg_id": torch.tensor(deg_id, dtype=torch.long),
                "severity": torch.tensor([severity], dtype=torch.float32)
            }
        except Exception as e:
            print(f"⚠️ Skipping corrupt image {index}: {e}")
            return self.__getitem__(random.randint(0, len(self.ids) - 1))