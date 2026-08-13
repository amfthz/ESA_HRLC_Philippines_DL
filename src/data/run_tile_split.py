# src/data/run_tile_split.py
import os
import json

from src.utils.config import load_config
from src.data.legend import load_legend
from src.data.indexing import list_label_tile_ids
from src.data.tile_stats import compute_all_tile_stats
from src.data.tile_split import create_tile_split

from typing import List, Optional
import numpy as np


def compute_class_weights_from_tile_stats(
    tile_stats: dict,
    train_tiles: list,
    epsilon: float = 1e-6,
    ignore_class_ids: Optional[List[int]] = None,
):
    """
    Compute class weights from tile-level class distributions
    using inverse square-root frequency.
    Weights are computed ONLY on training tiles.
    """
    train_vectors = np.array([tile_stats[t] for t in train_tiles])
    p_c = train_vectors.mean(axis=0)

    if ignore_class_ids is not None:
        for cid in ignore_class_ids:
            if cid < len(p_c):
                p_c[cid] = 0.0

    weights = np.zeros_like(p_c, dtype=np.float32)
    present = p_c > 0
    weights[present] = 1.0 / np.sqrt(p_c[present] + epsilon)

    if ignore_class_ids is not None:
        for cid in ignore_class_ids:
            if cid < len(weights):
                weights[cid] = 0.0

    if present.any():
        weights[present] /= weights[present].mean()

    return weights.tolist()


def run_tile_split(
    dataset_cfg: dict,
    legend_cfg_path: str = "configs/legend.yaml",
    force: bool = False,
    regions: Optional[List[str]] = None,
):
    cfg = dataset_cfg
    legend = load_legend(legend_cfg_path)

    train_cfg = load_config("configs/training.yaml")
    ignore_class_ids = train_cfg.get("loss", {}).get(
        "ignore_class_ids",
        [train_cfg.get("loss", {}).get("ignore_index", 0)]
    )

    # Resolve regions to process
    if regions is None:
        if "regions" in cfg.get("dataset", {}):
            regions = cfg["dataset"]["regions"]
        elif "region" in cfg.get("dataset", {}):
            regions = [cfg["dataset"]["region"]]
        else:
            raise ValueError("No 'region' or 'regions' key found in dataset config")

    # Normalize to list
    if isinstance(regions, str):
        regions = [regions]
    base_path = cfg["dataset"]["base_path"]
    splits_base = cfg["derived"]["splits_base_dir"]

    for region in regions:
        print(f"[run_tile_split] Region: {region}")

        root_gt = os.path.join(base_path, region, "ground_reference")
        root_labels = os.path.join(base_path, region, "labels")

        tile_ids = list_label_tile_ids(root_labels)
        if not tile_ids:
            print("  No label JSONs found, skipping.")
            continue

        region_dir = os.path.join(splits_base, region)
        os.makedirs(region_dir, exist_ok=True)

        stats_path = os.path.join(region_dir, cfg["derived"]["tile_stats_file"])
        split_path = os.path.join(region_dir, cfg["derived"]["tile_split_file"])

        if force or not os.path.exists(stats_path):
            print("  Computing tile statistics...")
            stats = compute_all_tile_stats(
                root_gt=root_gt,
                legend=legend,
                tile_ids=tile_ids,
                save_path=stats_path,
            )
        else:
            with open(stats_path) as f:
                stats = json.load(f)

        if not stats:
            print("  No valid tiles after stats. Skipping.")
            continue

        if force or not os.path.exists(split_path):
            print("  Creating train/val split...")
            create_tile_split(
                tile_stats=stats,
                n_clusters=cfg["tile_split"]["clustering"]["n_clusters"],
                val_fraction=cfg["tile_split"]["split"]["val_fraction"],
                random_state=cfg["tile_split"]["clustering"]["random_state"],
                save_path=split_path,
            )
        else:
            print("  Split already exists.")

        with open(split_path) as f:
            split = json.load(f)

        weights_path = os.path.join(region_dir, "class_weights.json")

        if force or not os.path.exists(weights_path):
            print("  Computing class weights (train tiles only)...")

            weights = compute_class_weights_from_tile_stats(
                tile_stats=stats,
                train_tiles=split["train_tiles"],
                ignore_class_ids=ignore_class_ids,
            )

            with open(weights_path, "w") as f:
                json.dump(
                    {
                        "method": "inverse_sqrt",
                        "epsilon": 1e-6,
                        "weights": weights,
                        "num_classes": len(weights),
                        "computed_on": "train_tiles_only",
                        "ignored_classes": ignore_class_ids,
                    },
                    f,
                    indent=2,
                )

            print(f"  Class weights saved to: {weights_path}")
        else:
            print("  Class weights already exist.")

        print("  DONE\n")
        print(f"Class weights : {weights_path}")


if __name__ == "__main__":

    dataset_cfg = load_config("configs/dataset.yaml")
    run_tile_split(
        dataset_cfg=dataset_cfg,
        legend_cfg_path="configs/legend.yaml",
        force=True,
        regions=None,
    )