# src/evaluation/io.py

import os
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from typing import Dict

# --------------------------------------------------
# ESA-CCI palette
# --------------------------------------------------
ESA_CCI_COLORS = {
    0: (0, 0, 0),           # NoData
    1: (0, 100, 0),         # Forest
    2: (85, 107, 47),       # Shrubland
    3: (124, 252, 0),       # Grassland
    4: (255, 255, 0),       # Cropland
    5: (0, 255, 255),       # Wetland
    6: (154, 205, 50),      # Lichens/mosses
    7: (210, 180, 140),     # Bareland
    8: (220, 20, 60),       # Built-up
    9: (0, 0, 255),         # Water
    10: (240, 248, 255),    # Ice/snow
}


# --------------------------------------------------
# Directory handling
# --------------------------------------------------
def prepare_output_dirs(base_dir: str, region: str) -> Dict[str, str]:
    """
    Create standard output directories for a region.

    Returns
    -------
    dict with keys: root, tif, plots
    """
    root = os.path.join(base_dir, region)
    tif_dir = os.path.join(root, "tif")
    plots_dir = os.path.join(root, "plots")

    os.makedirs(tif_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    return {
        "root": root,
        "tif": tif_dir,
        "plots": plots_dir,
    }


# --------------------------------------------------
# GeoTIFF utilities
# --------------------------------------------------
def save_like_gt(
    arr: np.ndarray,
    gt_path: str,
    out_path: str,
    dtype: str = "uint8",
):
    """
    Save single-band raster aligned to ground-truth geometry.
    """
    with rasterio.open(gt_path) as src:
        meta = src.meta.copy()

    resized = cv2.resize(
        arr.astype(dtype),
        (meta["width"], meta["height"]),
        interpolation=cv2.INTER_NEAREST,
    )

    meta.update(count=1, dtype=dtype)

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(resized, 1)


def save_probs_like_gt(
    probs: np.ndarray,
    gt_path: str,
    out_path: str,
):
    """
    Save probability cube (C, H, W) aligned to GT.
    """
    with rasterio.open(gt_path) as src:
        meta = src.meta.copy()

    C, H, W = probs.shape
    out = np.zeros((C, meta["height"], meta["width"]), dtype=np.float32)

    for c in range(C):
        out[c] = cv2.resize(
            probs[c],
            (meta["width"], meta["height"]),
            interpolation=cv2.INTER_NEAREST,
        )

    meta.update(count=C, dtype="float32")

    with rasterio.open(out_path, "w", **meta) as dst:
        for c in range(C):
            dst.write(out[c], c + 1)


# --------------------------------------------------
# Color utilities
# --------------------------------------------------
def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Convert class-id mask to RGB image.
    """
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, v in ESA_CCI_COLORS.items():
        rgb[mask == k] = v
    return rgb


# --------------------------------------------------
# Plot utilities
# --------------------------------------------------
def save_qualitative_plot(
    sar: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    out_path: str,
    water_id: int = 9,
):
    """
    Save qualitative plot:
    - SAR mean
    - Prediction (masked where GT is water)
    - Ground Truth
    """
    pred_vis = pred.copy()
    pred_vis[gt == water_id] = 0  # black where water in GT

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(sar, cmap="gray")
    axes[0].set_title("SAR mean")
    axes[0].axis("off")

    axes[1].imshow(colorize_mask(pred_vis))
    axes[1].set_title("Prediction")
    axes[1].axis("off")

    axes[2].imshow(colorize_mask(gt))
    axes[2].set_title("Ground Truth")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------
# JSON utilities
# --------------------------------------------------
def save_metrics(metrics: Dict, out_path: str):
    """
    Save metrics dictionary as JSON.
    """
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)