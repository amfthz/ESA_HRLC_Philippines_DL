# src/evaluation/core.py

import torch
import torch.nn.functional as F
from typing import List, Tuple

from src.data.patching import patch_tensors


def confidence_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    Compute confidence map from probabilities using normalized entropy.

    probs: (N, C, H, W)
    returns: (N, 1, H, W) in [0,1], higher = more confident
    """
    C = probs.shape[1]
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1, keepdim=True)
    entropy_norm = entropy / torch.log(torch.tensor(C, device=probs.device))
    confidence = 1.0 - entropy_norm
    return torch.clamp(confidence, 0.0, 1.0)


def gaussian_weight_window(patch_size, sigma_scale=0.125):
    ph, pw = patch_size
    y = torch.arange(ph, dtype=torch.float32) - (ph - 1) / 2
    x = torch.arange(pw, dtype=torch.float32) - (pw - 1) / 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    sigma_y = ph * sigma_scale
    sigma_x = pw * sigma_scale
    gauss = torch.exp(-0.5 * ((yy / sigma_y) ** 2 + (xx / sigma_x) ** 2))
    gauss /= gauss.max()
    return gauss.unsqueeze(0)  # shape (1, ph, pw)


def cosine_weight_window(patch_size):
    ph, pw = patch_size
    y = torch.arange(ph, dtype=torch.float32)
    x = torch.arange(pw, dtype=torch.float32)
    wy = 0.5 * (1 - torch.cos(2 * torch.pi * y / (ph - 1)))
    wx = 0.5 * (1 - torch.cos(2 * torch.pi * x / (pw - 1)))
    w = wy.unsqueeze(1) * wx.unsqueeze(0)
    w /= w.max()
    return w.unsqueeze(0)  # shape (1, ph, pw)


@torch.inference_mode()
def extract_patches_with_coords(
    x: torch.Tensor,
    patch_size: Tuple[int, int],
    stride: int,
):
    """
    Extract patches and their top-left coordinates.

    Parameters
    ----------
    x : torch.Tensor
        Input tile tensor of shape (1, T, H, W)
    patch_size : (ph, pw)
    stride : int

    Returns
    -------
    patches : torch.Tensor
        Shape (N, T, ph, pw)
    coords : List[(y, x)]
        Top-left coordinates of each patch
    tile_hw : (H, W)
    """
    assert x.ndim == 4, "Input must be (1, T, H, W)"

    _, _, H, W = x.shape
    ph, pw = patch_size

    patches = patch_tensors(x, patch_size, stride)  # (N, T, ph, pw)

    coords: List[Tuple[int, int]] = []
    for yy in range(0, H - ph + 1, stride):
        for xx in range(0, W - pw + 1, stride):
            coords.append((yy, xx))

    assert len(coords) == patches.shape[0], (
        f"Mismatch between coords ({len(coords)}) and patches ({patches.shape[0]})"
    )

    return patches, coords, (H, W)


@torch.inference_mode()
def reconstruct_from_patches_weighted(
    probs_patches: torch.Tensor,
    coords: List[Tuple[int, int]],
    tile_hw: Tuple[int, int],
    patch_size: Tuple[int, int],
    weight_window: torch.Tensor,
    confidence_patches: torch.Tensor | None = None,
):
    """
    Reconstruct full-tile probability map from patch probabilities
    using weighted blending in overlapping regions.

    Parameters
    ----------
    probs_patches : torch.Tensor
        Shape (N, C, ph, pw)
    coords : list of (y, x)
    tile_hw : (H, W)
    patch_size : (ph, pw)
    weight_window : torch.Tensor
        Shape (1, ph, pw)
    confidence_patches : torch.Tensor or None
        Shape (N, 1, ph, pw), optional confidence weights per patch

    Returns
    -------
    probs_tile : torch.Tensor
        Shape (C, H, W)
    """
    N, C, ph, pw = probs_patches.shape
    H, W = tile_hw

    device = probs_patches.device
    weight_window = weight_window.to(device)

    prob_sum = torch.zeros((C, H, W), device=device)
    prob_cnt = torch.zeros((1, H, W), device=device)

    for i, (yy, xx) in enumerate(coords):
        w = weight_window
        if confidence_patches is not None:
            w = w * confidence_patches[i]

        prob_sum[:, yy:yy + ph, xx:xx + pw] += probs_patches[i] * w
        prob_cnt[:, yy:yy + ph, xx:xx + pw] += w

    prob_cnt = torch.clamp(prob_cnt, min=1.0)

    return prob_sum / prob_cnt


@torch.inference_mode()
def infer_tile_patchwise(
    model: torch.nn.Module,
    x: torch.Tensor,
    patch_size: Tuple[int, int],
    stride: int,
    blend_mode: str = "gaussian",
    temperature: float = 1.0,
):
    """
    Perform patch-based inference on a single tile and reconstruct it.

    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model
    x : torch.Tensor
        Input tile of shape (1, T, H, W)
    patch_size : (ph, pw)
    stride : int
    blend_mode : str, optional
        Blending mode: 'gaussian', 'cosine', or 'uniform' (default: 'gaussian')
    temperature : float, optional
        Temperature scaling factor for logits (default: 1.0)

    Returns
    -------
    probs_tile : torch.Tensor
        Shape (C, H, W)
    pred_tile : torch.Tensor
        Shape (H, W)
    """
    patches, coords, tile_hw = extract_patches_with_coords(
        x, patch_size, stride
    )

    # Patch inference
    logits_p = model(patches)
    logits_p = logits_p / temperature
    probs_p = F.softmax(logits_p, dim=1)
    confidence_p = confidence_from_probs(probs_p)

    # Create weight window
    if blend_mode == "gaussian":
        weight_window = gaussian_weight_window(patch_size)
    elif blend_mode == "cosine":
        weight_window = cosine_weight_window(patch_size)
    elif blend_mode == "uniform":
        weight_window = torch.ones((1, patch_size[0], patch_size[1]), dtype=torch.float32)
    else:
        raise ValueError(f"Unknown blend_mode: {blend_mode}")

    # Reconstruct full tile
    probs_tile = reconstruct_from_patches_weighted(
        probs_p,
        coords,
        tile_hw,
        patch_size,
        weight_window,
        confidence_patches=confidence_p,
    )

    pred_tile = torch.argmax(probs_tile, dim=0)

    return probs_tile, pred_tile