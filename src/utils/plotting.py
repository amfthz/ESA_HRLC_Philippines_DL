# src/utils/plotting.py

import numpy as np
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import torch
from typing import Optional


def plot_samples(
    dataset,
    num_samples: int = 4,
    seed: Optional[int] = None,
    cmap_input: str = "gray",
    cmap_gt: str = "tab20",
    save_dir: Optional[str] = None,
):
    """
    Plot random samples from a PyTorch Dataset for sanity check.

    Assumes dataset[i] returns:
      - x: Tensor (T, H, W)
      - y: Tensor (1, H, W) or (H, W)

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset instance.
    num_samples : int
        Number of random samples to plot.
    seed : int, optional
        Random seed for reproducibility.
    cmap_input : str
        Colormap for input (SAR).
    cmap_gt : str
        Colormap for ground truth.
    save_dir : str, optional
        Directory where the sanity-check figure will be saved. If None, defaults to "outputs/sanity_checks".
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dataset_len = len(dataset)
    if dataset_len == 0:
        raise ValueError("Dataset is empty.")

    indices = rng.choice(dataset_len, size=min(num_samples, dataset_len), replace=False)

    n_cols = 4  # input + GT pairs per row (2 columns per sample)
    n_rows = int(np.ceil(len(indices) / n_cols))
    plt.figure(figsize=(8, 2.2 * n_rows))

    for row, idx in enumerate(indices):
        x, y = dataset[idx]

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()

        # x can be (T,H,W) or (P,T,H,W)
        if x.ndim == 4:
            # patched case: select a random patch, then average over time
            p_idx = rng.integers(0, x.shape[0])
            x_patch = x[p_idx]          # (T,H,W)
            x_show = x_patch.mean(axis=0)
        else:
            # unpatched case: directly average over time
            x_show = x.mean(axis=0)

        # y can be (1,H,W), (H,W), (P,1,H,W) or (P,H,W)
        if y.ndim == 4:
            y_patch = y[p_idx]
            y_show = y_patch[0] if y_patch.ndim == 3 else y_patch
        elif y.ndim == 3:
            y_show = y[0]
        else:
            y_show = y

        # ---- INPUT ----
        base_idx = row * 2
        plt.subplot(n_rows, n_cols * 2, base_idx + 1)
        plt.imshow(x_show, cmap=cmap_input)
        plt.axis("off")

        # ---- GT ----
        plt.subplot(n_rows, n_cols * 2, base_idx + 2)
        plt.imshow(y_show, cmap=cmap_gt, vmin=0)
        plt.axis("off")

    # ---- OUTPUT PATH ----
    if save_dir is None:
        save_dir = os.path.join("outputs", "sanity_checks")

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "dataset_sanity_check.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PLOTTING] Sanity check saved to: {out_path}")