# src/data/tile_split.py
import json
import numpy as np
from typing import List, Dict
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit


def create_tile_split(
    tile_stats: Dict[str, List[float]],
    n_clusters: int,
    val_fraction: float,
    random_state: int,
    save_path: str,
):
    """
    Cluster + stratified split using ONLY tiles present in tile_stats.
    """

    tile_ids = sorted(tile_stats.keys())
    if len(tile_ids) == 0:
        raise RuntimeError("No valid tiles for splitting.")

    P = np.array([tile_stats[t] for t in tile_ids], dtype=np.float32)

    # fallback: too few tiles
    if len(tile_ids) < max(n_clusters, 5):
        rng = np.random.default_rng(random_state)
        perm = rng.permutation(len(tile_ids))
        n_val = int(np.ceil(val_fraction * len(tile_ids)))

        val_tiles = [tile_ids[i] for i in perm[:n_val]]
        train_tiles = [tile_ids[i] for i in perm[n_val:]]
    else:
        km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_ids = km.fit_predict(P)

        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_fraction,
            random_state=random_state,
        )
        idx = np.arange(len(tile_ids))
        tr, va = next(sss.split(idx, cluster_ids))

        train_tiles = [tile_ids[i] for i in tr]
        val_tiles = [tile_ids[i] for i in va]

    split = {
        "n_tiles": len(tile_ids),
        "n_clusters": n_clusters,
        "val_fraction": val_fraction,
        "train_tiles": train_tiles,
        "val_tiles": val_tiles,
    }

    with open(save_path, "w") as f:
        json.dump(split, f, indent=2)

    print(
        f"[tile_split] tiles: {len(tile_ids)} | "
        f"train: {len(train_tiles)} | val: {len(val_tiles)}"
    )