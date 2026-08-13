# src/data/patching.py

import torch
import torch.nn.functional as F

def patch_tensors(
    tensor: torch.Tensor,
    patch_size: tuple,
    stride: int,
) -> torch.Tensor:
    """
    Extract patches from a batch of tensors using torch.unfold (fast, vectorized).

    Parameters
    ----------
    tensor : torch.Tensor
        Shape (B, T, H, W)
    patch_size : tuple
        (ph, pw)
    stride : int

    Returns
    -------
    torch.Tensor
        Shape (B * P, T, ph, pw)
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    B, T, H, W = tensor.shape
    ph, pw = patch_size

    # (B, T, H, W) → (B*T, 1, H, W)
    x = tensor.reshape(B * T, 1, H, W)

    # unfold: (B*T, ph*pw, P)
    patches = F.unfold(
        x,
        kernel_size=(ph, pw),
        stride=stride,
    )

    # P = number of patches per image
    P = patches.shape[-1]

    # (B*T, ph*pw, P) → (B, T, P, ph, pw)
    patches = patches.transpose(1, 2)
    patches = patches.reshape(B, T, P, ph, pw)

    # (B, T, P, ph, pw) → (B, P, T, ph, pw)
    patches = patches.permute(0, 2, 1, 3, 4)

    # (B, P, T, ph, pw) → (B*P, T, ph, pw)
    patches = patches.reshape(B * P, T, ph, pw)

    return patches