import torch
import torch.nn as nn

from .clip import CLIP

def convert_weights(model: nn.Module) -> None:
  """Cast all applicable parameters in *model* to fp16 in-place."""

  def _to_fp16(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
      module.weight.data = module.weight.data.half()
      if module.bias is not None:
        module.bias.data = module.bias.data.half()

    if isinstance(module, nn.MultiheadAttention):
      for attr_name in [
        *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
        "in_proj_bias",
        "bias_k",
        "bias_v",
      ]:
        tensor = getattr(module, attr_name)
        if tensor is not None:
            tensor.data = tensor.data.half()

    for param_name in ["text_projection", "proj"]:
      if hasattr(module, param_name):
        attr = getattr(module, param_name)
        if attr is not None:
          attr.data = attr.data.half()

  model.apply(_to_fp16)
 
 
# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
 
def build_model(state_dict: dict) -> CLIP:
  """Reconstruct a CLIP model from a state dict and return it in eval mode."""

  vit = "visual.proj" in state_dict

  if vit:
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([
      k for k in state_dict
      if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
    ])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
  else:
    # FIX: original file was truncated here — the tuple(counts) line and
    # all subsequent lines were missing, making build_model unusable for
    # ResNet-backed CLIP checkpoints.
    counts: list = [
        len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}")))
        for b in [1, 2, 3, 4]
    ]
    vision_layers = tuple(counts)
    vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
    output_width = round(
        (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
    )
    vision_patch_size = None
    assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
    image_resolution = output_width * 32

  embed_dim         = state_dict["text_projection"].shape[1]
  context_length    = state_dict["positional_embedding"].shape[0]
  vocab_size        = state_dict["token_embedding.weight"].shape[0]
  transformer_width = state_dict["ln_final.weight"].shape[0]
  transformer_heads = transformer_width // 64
  transformer_layers = len(set(
      k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")
  ))

  model = CLIP(
    embed_dim,
    image_resolution, vision_layers, vision_width, vision_patch_size,
    context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
  )

  # Remove non-parameter metadata keys before loading
  for key in ["input_resolution", "context_length", "vocab_size"]:
    state_dict.pop(key, None)

  convert_weights(model)
  model.load_state_dict(state_dict)
  return model.eval()