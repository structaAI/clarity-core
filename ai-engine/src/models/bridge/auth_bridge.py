import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Any

class AuthBridge(nn.Module):
  def __init__(self, input_dim: int, output_dim: int)-> None:
    super().__init__()
    self.linear = nn.Linear(input_dim, output_dim)
  
  def forward(self, x: torch.Tensor)-> Any:\
    pass