# Imports
import torch
import torch.nn as nn
import math
from typing import Tuple


"""
TimeStep Embessing Module

--Parameters--
- @hidden_size: int: Size of the Hidden Layer
- @frequency_embedding_size: int: Size of the Frequency Embedding (Default: 256)

--Architecture Overview--
1. Frequency Embedding: Generates Sinusoidal Embeddings based on Time Steps
2. MLP: Multi-Layer Perceptron to process Frequency Embeddings

--Returns--
- torch.Tensor: Time Step Embeddings of shape (Batch Size, hidden_size)

Over here, we first create sinusoidal embeddings for the time steps
Done with the help of sine and cosine functions at different frequencies

i.e [sin(timestep / 10000^(2*i/dim)), cos(timestep / 10000^(2*i/dim))]
This is inspired from the positional encoding used in Transformer models

Note: 10000 is a scaling factor that helps in spreading out the embeddings (Decided via Experiments)
"""
class TimeStepEmbedding(nn.Module):
  def __init__(self, hidden_size: int, frequency_embedding_size: int = 256, num_degradation_types: int = 5)-> None:
    super().__init__()

    # Initializing Parameters for Instance
    self.hidden_size = hidden_size
    self.frequency_embedding_size = frequency_embedding_size

    # Defining the Multi-Layer Perceptron (MLP)
    """
    --Components--
    1. Linear Layer
    2. SiLU ACtivation Function
    3. Final Mapping Linear Layer
    """
    self.mlp: nn.Sequential = nn.Sequential(
      nn.Linear(self.frequency_embedding_size, self.hidden_size),
      nn.SiLU(),
      nn.Linear(self.hidden_size, self.hidden_size),
    )

    self.type_embeddings = nn.Embedding(num_degradation_types, hidden_size)

    # 3. Component: Severity (s_σ)
    # Projects a scalar severity value [0, 1] into the embedding space.
    self.severity_encoder = nn.Sequential(
        nn.Linear(1, hidden_size // 4),
        nn.SiLU(),
        nn.Linear(hidden_size // 4, hidden_size)
    )

    # 4. Component: Tripartite Fusion via Cross-Attention
    # Allows the model to attend to the most relevant conditioning signal.
    self.fusion_attention = nn.MultiheadAttention(
        hidden_size, num_heads=8, batch_first=True
    )
  
  def forward(self, x: torch.Tensor)-> torch.Tensor:
    half_dim = self.frequency_embedding_size // 2

    # Defining the freqency Formula as seen above
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half_dim, device=x.device)/half_dim)
    args = x[:, None].float() * freqs[None, :]

    # Concatenating the Sine and Cosine Embeddings
    x_freq = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    # Return the MLP output
    return self.mlp(x_freq)


"""
Patch Embedding Module

--Parameters--
- @patch_size: int: Size of each Patch (default: 2)
- @no_of_in_channels: int: Number of Input Channels (default: 4)
- @embed_dim: int: Dimension of the Embedding (default: 768)

--Returns--
- Tuple[torch.Tensor, Tuple[int, int]]: 
  1. Patch Embeddings of shape (Batch Size, Number of Patches, embed_dim)
  2. Original Height and Width of the feature map after patching

Over here, we first create a linear projection of the input spatial manifold. 
Theoretical Process:
1. Spatial Partitioning: The 2D input latent x: (B, C, H, W) is 
   partitioned into non-overlapping patches of size PxP. This results in a 
   feature map of size (B, C, H/P, W/P)
2. Dimensional Projection: Each patch is flattened and projected into an 
   embedding space of dimension $D$ using a 2D convolution. This transforms the 
   local spatial information into a localized feature vector.
3. Sequence Serialization: The spatial grid is flattened into a 1D sequence of length 
   L=(H/P)*(W/P). This format allows the Swin-Transformer blocks to 
   process the image as a sequence for Autoregressive-like structural 
   modeling.
4. Normalization: LayerNorm is applied to the sequence to stabilize the 
   distribution before it enters the hierarchical window blocks.
"""
class SwinPatchEmbed(nn.Module):
  def __init__(self, patch_size: int = 2, no_of_in_channels: int = 4, embed_dim: int = 768)-> None:
    super().__init__()

    # Defining the Patch Embedding Layers and Layer Normalization
    self.projection: nn.Conv2d = nn.Conv2d(no_of_in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    self.norm: nn.LayerNorm = nn.LayerNorm(embed_dim)
  
  def forward(self, x: torch.Tensor)-> Tuple[torch.Tensor, Tuple[int, int]]:
    x = self.projection(x)
    _, _, H, W = x.shape # Batch Size (B), Channels (C), Height (H), Width (W)

    # Flattening and Transposing the Tensor to get Patch Embeddings
    x = x.flatten(2).transpose(1, 2)

    # Returning the Normalized Patch Embeddings along with Original Height and Width
    return self.norm(x), (H, W)
