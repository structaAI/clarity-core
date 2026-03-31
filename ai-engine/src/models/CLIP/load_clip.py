import os
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv('MODEL_NAME', 'google/siglip-so400m-patch14-384')
SAVE_PATH = os.getenv('CLIP_MODEL_SAVE_PATH', 'D:/Structa/claritycore/ai-engine/models/CLIP/saved_models/siglip-so400m-patch14-38')

def force_save():
  print(f"Force-saving {MODEL_NAME} to {SAVE_PATH}...")
  
  model = AutoModel.from_pretrained(MODEL_NAME)
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  
  os.makedirs(SAVE_PATH, exist_ok=True)
  
  model.save_pretrained(SAVE_PATH)
  tokenizer.save_pretrained(SAVE_PATH)
  
  print(f"Success! Check {SAVE_PATH} for 'model.safetensors' and 'config.json'.")

if __name__ == "__main__":
  force_save()