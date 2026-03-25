import torch
import numpy as np
from torch import nn
from typing import Tuple, Union

from .modified_resnet import ModifiedResNet      # FIX: relative import
from .vision_transformer import VisionTransformer  # FIX: relative import
from .transformer import Transformer              # FIX: relative import
from .layer_norm import LayerNorm                 # FIX: relative import


class CLIP(nn.Module):
  def __init__(
    self,
    embed_dim: int,
    # Vision
    image_resolution: int,
    vision_layers: Union[Tuple[int, int, int, int], int],
    vision_width: int,
    vision_patch_size: int,
    # Text
    context_length: int,
    vocab_size: int,
    transformer_width: int,
    transformer_heads: int,
    transformer_layers: int,
  ) -> None:
    super().__init__()
    self.context_length = context_length

    if isinstance(vision_layers, (tuple, list)):
      vision_heads = vision_width * 32 // 64
      self.visual: Union[ModifiedResNet, VisionTransformer] = ModifiedResNet(
        layers=vision_layers,
        output_dim=embed_dim,
        heads=vision_heads,
        input_resolution=image_resolution,
        width=vision_width,
      )
    else:
      vision_heads = vision_width // 64
      self.visual = VisionTransformer(
        input_resolution=image_resolution,
        patch_size=vision_patch_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_heads,
        output_dim=embed_dim,
      )

    self.transformer = Transformer(
      width=transformer_width,
      layers=transformer_layers,
      heads=transformer_heads,
      attn_mask=self.build_attention_mask(),
    )

    self.vocab_size = vocab_size
    self.token_embedding = nn.Embedding(vocab_size, transformer_width)
    self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
    self.ln_final = LayerNorm(transformer_width)
    self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
    self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    self.initialize_parameters()

  # ------------------------------------------------------------------
  # Initialisation
  # ------------------------------------------------------------------

  def initialize_parameters(self) -> None:
    nn.init.normal_(self.token_embedding.weight, std=0.02)
    nn.init.normal_(self.positional_embedding, std=0.01)

    if isinstance(self.visual, ModifiedResNet):
      if self.visual.attnpool is not None:
        std = self.visual.attnpool.c_proj.in_features ** -0.5
        for proj in [
          self.visual.attnpool.q_proj,
          self.visual.attnpool.k_proj,
          self.visual.attnpool.v_proj,
          self.visual.attnpool.c_proj,
        ]:
          nn.init.normal_(proj.weight, std=std)

      # FIX: zero-init bn3.weight in every ResNet stage.
      # This was present in the reference notebook but missing from the
      # modular version, causing different training behaviour.
      for layer in [
        self.visual.layer1,
        self.visual.layer2,
        self.visual.layer3,
        self.visual.layer4,
      ]:
        for name, param in layer.named_parameters():
          if name.endswith("bn3.weight"):
            nn.init.zeros_(param)

    proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
    attn_std = self.transformer.width ** -0.5
    fc_std   = (2 * self.transformer.width) ** -0.5
    for block in self.transformer.resblocks:
      nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
      nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
      nn.init.normal_(block.mlp.c_fc.weight,  std=fc_std)
      nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    if self.text_projection is not None:
      nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

  # ------------------------------------------------------------------
  # Helpers
  # ------------------------------------------------------------------

  def build_attention_mask(self) -> torch.Tensor:
    """Causal mask: upper triangle filled with -inf, diagonal and below = 0."""
    mask = torch.empty(self.context_length, self.context_length)
    mask.fill_(float("-inf"))
    mask.triu_(1)
    return mask

  @property
  def dtype(self) -> torch.dtype:
    # FIX: both ModifiedResNet and VisionTransformer expose conv1, so this
    # property is safe for either visual backbone.
    return self.visual.conv1.weight.dtype

  # ------------------------------------------------------------------
  # Encoding
  # ------------------------------------------------------------------

  def encode_image(self, image: torch.Tensor) -> torch.Tensor:
    return self.visual(image.type(self.dtype))

  def encode_text(self, text: torch.Tensor) -> torch.Tensor:
    x = self.token_embedding(text).type(self.dtype)   # [B, n_ctx, d_model]
    x = x + self.positional_embedding.type(self.dtype)
    x = x.permute(1, 0, 2)                            # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)                            # LND -> NLD
    x = self.ln_final(x).type(self.dtype)
    # Take features from the EOT token (highest token index in each sequence)
    x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
    return x

  # ------------------------------------------------------------------
  # Forward
  # ------------------------------------------------------------------

  def forward(
    self, image: torch.Tensor, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    image_features = self.encode_image(image)
    text_features  = self.encode_text(text)

    # L2-normalise both modalities
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features  = text_features  / text_features.norm(dim=1, keepdim=True)

    # Scaled cosine similarity
    logit_scale = self.logit_scale.exp()
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text  = logits_per_image.t()

    return logits_per_image, logits_per_text   # [global_B, global_B]