# Imports
import torch
from typing import Tuple


def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, int, int]:
    """
    Partition input tensor into non-overlapping local windows.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, H, W, C).
    window_size : int
        Side length of each square window.
        Both H and W must be divisible by window_size.

    Returns
    -------
    windows : torch.Tensor
        Partitioned windows of shape (num_windows * B, window_size, window_size, C).
    H : int
        Original height of the input tensor.      ← FIX: docstring previously said
    W : int                                            "partition height (p_H)" which
        Original width of the input tensor.            was wrong and misleading.
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, H, W


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    """
    Reconstruct the original spatial tensor from partitioned windows.

    Parameters
    ----------
    windows : torch.Tensor
        Partitioned windows of shape (num_windows * B, window_size, window_size, C).
    window_size : int
        Side length of each square window.
    H : int
        Original height of the tensor (before partitioning).
    W : int
        Original width of the tensor (before partitioning).

    Returns
    -------
    torch.Tensor
        Reconstructed tensor of shape (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    C = windows.shape[-1]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x