import torch
from torch import nn
from residual_attention_block import ResidualAttentionBlock

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers