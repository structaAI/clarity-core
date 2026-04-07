import torch
import torch.nn as nn
import numpy as np
import gradio as gr
from PIL import Image
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.swin_dit.backbone import SwinDiT
from src.models.bridge.auth_bridge import AuthBridge
from src.training.mini_model_integrated_training import load_config

# --- SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../config/model_config.yaml")
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "../../experiments/auth_swindiff_demo.pt")

config = load_config(CONFIG_PATH)

def load_demo_models():
  bridge = AuthBridge(input_dim=1152, output_dim=config.model.embed_dim).to(device)
  model = SwinDiT(config.model).to(device)
  
  if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    bridge.load_state_dict(checkpoint['bridge_state_dict'])
    print("Checkpoint loaded successfully!")
  else:
    print("No checkpoint found. Demo will use untrained weights.")
  
  model.eval()
  bridge.eval()
  return bridge, model

BRIDGE, MODEL = load_demo_models()

@torch.no_grad()
def restore_image(input_img, prompt, deg_type):
  if input_img is None: return None
  
  # Pre-process
  original_size = input_img.size
  img = input_img.resize((512, 512))
  img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
  img_tensor = img_tensor.unsqueeze(0).to(device)
  
  # Demo logic: Static timestep & dummy clip
  t = torch.tensor([500]).to(device) 
  dummy_clip = torch.randn(1, 1152).to(device) 
  deg_map = {"Blur": 0, "Noise": 1, "Low-Res": 2}
  d_id = torch.tensor([deg_map[deg_type]]).to(device)
  sev = torch.tensor([[0.5]]).to(device)

  # Inference
  cond = BRIDGE(t_emb=t, clip_emb=dummy_clip, deg_type=d_id, severity=sev)
  output_tensor = MODEL(img_tensor, cond)
  
  # Post-process
  out_np = output_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
  out_np = (out_np * 255).clip(0, 255).astype(np.uint8)
  
  return Image.fromarray(out_np).resize(original_size)

# --- UI ---
with gr.Blocks(title="Auth-SwinDiff Demo") as demo:
  gr.Markdown("# Auth-SwinDiff: Generative Restoration")
  with gr.Row():
    with gr.Column():
      in_img = gr.Image(type="pil", label="Input")
      in_prompt = gr.Textbox(label="Prompt")
      in_mode = gr.Dropdown(["Blur", "Noise", "Low-Res"], value="Blur", label="Mode")
      btn = gr.Button("Restore", variant="primary")
    with gr.Column():
      out_img = gr.Image(type="pil", label="Output")

  btn.click(restore_image, [in_img, in_prompt, in_mode], out_img)

if __name__ == "__main__":
  demo.launch(share=True)