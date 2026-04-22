import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import yaml
from types import SimpleNamespace
from transformers import SiglipTokenizer, SiglipTextModel
from dotenv import load_dotenv


load_dotenv(".env.local")
# Internal Imports
from src.models.swin_dit.backbone import SwinDiT
from src.models.bridge.auth_bridge import AuthBridge

# 1. CONFIG LOADER
def load_config(config_path):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
        return d
    return dict_to_namespace(config_dict)

# 2. MODEL INITIALIZATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = load_config(r"D:\Structa\claritycore\ai-engine\src\configs\swin_dit_config.yaml")

print("⏳ Loading Local SigLIP so400m Reasoning Engine...")
# Local Path for SigLIP
siglip_path = r"D:\Structa\claritycore\ai-engine\src\models\CLIP\saved_models\siglip-so400m-patch14-384"

tokenizer = SiglipTokenizer.from_pretrained(siglip_path)
text_encoder = SiglipTextModel.from_pretrained(siglip_path).to(device) # type: ignore

# Load Structa Labs Restoration Models
model = SwinDiT(config).to(device)
bridge = AuthBridge(input_dim=768, output_dim=768).to(device)

# Load Trained Weights
ckpt_path = r"D:\Structa\claritycore\ai-engine\experiments\auth_swindiff_final.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    bridge.load_state_dict(ckpt['bridge_state_dict'])
    print(f"Weights loaded from {ckpt_path}")
else:
    print("Checkpoint not found. Initializing with random weights.")

model.eval()
bridge.eval()
text_encoder.eval()

# 3. INFERENCE ENGINE
def restore_image(input_img, prompt, timestep):
    if input_img is None: return None
    
    # Pre-process: Resize to 256 for Blackwell VRAM stability
    img = Image.fromarray(input_img).convert('RGB').resize((256, 256))
    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    
    # 4-Channel Padding
    zeros = torch.zeros((1, 1, 256, 256), device=device)
    img_4ch = torch.cat([img_tensor, zeros], dim=1)
    
    with torch.no_grad():
        # Encode Text using SigLIP
        inputs = tokenizer([prompt], padding=True, return_tensors="pt").to(device)
        # so400m output is [1, 1152]
        full_features = text_encoder(**inputs).last_hidden_state.mean(dim=1) 
        
        # PROJECTION: Slice or Pool 1152 down to 768 to match AuthBridge
        text_features = full_features[:, :768] 
        
        # Prepare Time Condition
        t_raw = torch.tensor([timestep], device=device).float().unsqueeze(1).repeat(1, 768)

        # AuthBridge Handshake
        cond = bridge(x=text_features, cond=t_raw)
        
        # Forward through SwinDiT
        output = model(img_4ch, precomputed_cond=cond.view(1, 768))
        
        # Post-process
        output_rgb = output[:, :3, :, :].clamp(0, 1).cpu().squeeze(0).permute(1, 2, 0).numpy()
        return (output_rgb * 255).astype(np.uint8)

# 4. GRADIO INTERFACE
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# Structa Labs | ClarityCore")
    gr.Markdown("### Auth-SwinDiff Reasoning-Based Restoration")
    
    with gr.Tab("Restoration Engine"):
        with gr.Row():
            with gr.Column():
                in_img = gr.Image(label="Input Image")
                in_text = gr.Textbox(label="Reasoning Prompt", value="restore sharp textures, eliminate noise")
                in_time = gr.Slider(0, 1000, value=500, label="Denoising Strength")
                btn = gr.Button("Process with Auth-SwinDiff", variant="primary")
            
            with gr.Column():
                out_img = gr.Image(label="ClarityCore Result")
        
        btn.click(restore_image, inputs=[in_img, in_text, in_time], outputs=out_img)

    with gr.Tab("System Info"):
        gr.Markdown(f"""
        | Parameter | Value |
        | :--- | :--- |
        | **Vision-Language Model** | SigLIP-so400m (Local) |
        | **Backbone** | Swin Transformer DiT |
        | **GPU** | NVIDIA RTX 5070 Ti |
        | **Precision** | bfloat16 Mixed |
        """)

if __name__ == "__main__":
  demo.launch(share=True)