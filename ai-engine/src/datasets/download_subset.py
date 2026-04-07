import fiftyone as fo
import fiftyone.zoo as foz
import os
import shutil

print("📥 Downloading 500 Golden Samples from COCO...")
dataset = foz.load_zoo_dataset(
  "coco-2017",
  split="validation", 
  max_samples=500,
  shuffle=True,
)

# 2. Project Paths
base_dir = r"D:\Structa\claritycore\ai-engine\data"
img_dir = os.path.join(base_dir, "train2017")
ann_dir = os.path.join(base_dir, "annotations")

os.makedirs(img_dir, exist_ok=True)
os.makedirs(ann_dir, exist_ok=True)

# 3. Export as COCO format
print("📂 Exporting to project folders...")
dataset.export(
  export_dir=base_dir,
  dataset_type="fiftyone.types.COCODetectionDataset",
  label_types=["captions"], 
)

# 4. Move and Rename to match your code's logic
if os.path.exists(os.path.join(base_dir, "data")):
  for f in os.listdir(os.path.join(base_dir, "data")):
    target_path = os.path.join(img_dir, f)
    if os.path.exists(target_path): os.remove(target_path)
    shutil.move(os.path.join(base_dir, "data", f), img_dir)

if os.path.exists(os.path.join(base_dir, "labels.json")):
  shutil.move(os.path.join(base_dir, "labels.json"), 
              os.path.join(ann_dir, "captions_train2017.json"))

print(f"✅ Ready! 500 images are in {img_dir}")