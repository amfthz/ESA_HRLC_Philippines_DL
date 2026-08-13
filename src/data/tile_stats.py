# src/data/tile_stats.py
import os
import json
import numpy as np
from typing import Dict, List, Optional

from src.utils.raster_io import read_raster
from src.data.preprocessing import nan_to_zero, remap_mask_values
from src.data.legend import Legend


def compute_tile_class_distribution(
    gt_path: str,
    legend: Legend,
) -> np.ndarray:
    gt, _ = read_raster(gt_path)
    if gt.ndim == 3:
        gt = gt[0]

    gt = nan_to_zero(gt)
    gt = remap_mask_values(gt, legend.raw_to_class)

    flat = gt.reshape(-1)
    valid = flat != legend.ignore_index
    flat = flat[valid]

    p = np.zeros(legend.num_classes, dtype=np.float32)
    if flat.size == 0:
        return p

    for c in range(legend.num_classes):
        p[c] = np.sum(flat == c)

    return p / p.sum()


def compute_all_tile_stats(
    root_gt: str,
    legend: Legend,
    tile_ids: List[str],
    save_path: Optional[str] = None,
) -> Dict[str, List[float]]:
    """
    Compute stats ONLY for tile_ids provided (from labels).
    Missing GT → tile ignored.
    """

    stats = {}
    skipped = 0

    for tid in tile_ids:
        gt_name = f"{tid.replace('_', '_GT_', 1)}.tif"
        gt_path = os.path.join(root_gt, gt_name)

        if not os.path.isfile(gt_path):
            skipped += 1
            continue

        try:
            p = compute_tile_class_distribution(gt_path, legend)
            stats[tid] = p.tolist()
        except Exception:
            skipped += 1

    if save_path:
        with open(save_path, "w") as f:
            json.dump(stats, f, indent=2)

    print(
        f"[tile_stats] valid: {len(stats)} | skipped: {skipped}"
    )

    return stats