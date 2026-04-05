import torch
import sys
from pathlib import Path
from PIL import Image
from transformers import AutoModel, AutoProcessor

current_file = Path(__file__).resolve()
src_path = current_file.parents[2] 
if str(src_path) not in sys.path:
  sys.path.insert(0, str(src_path))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = r"D:\Structa\claritycore\ai-engine\src\models\CLIP\saved_models\siglip-so400m-patch14-384"

def calculate_siglip_score(image_path, prompt):
  print(f"\033[1mEvaluating SigLIP Alignment\033[0m")
  print(f"Loading weights from: {MODEL_PATH}")

  processor = AutoProcessor.from_pretrained(MODEL_PATH)
  model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE).eval()

  try:
    image = Image.open(image_path).convert("RGB")
  except FileNotFoundError:
    print(f"{image_path} not found, using dummy data.")
    image = Image.new('RGB', (384, 384), color = (73, 109, 137))

  inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(DEVICE)

  # 3. Inference
  with torch.no_grad():
    outputs = model(**inputs)

    image_features = outputs.image_embeds
    text_features = outputs.text_embeds

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    score = torch.matmul(image_features, text_features.T).item()

  mem = torch.cuda.max_memory_allocated() / (1024**2)
  
  print(f"{'-'*40}")
  print(f"Prompt:      {prompt}")
  print(f"CLIP Score:  {score:.4f} (SigLIP Logits)")
  print(f"VRAM Peak:   {mem:.2f} MB")
  
  return score

if __name__ == "__main__":
  calculate_siglip_score("demo.png", "A high-quality denoised image of a sunset")