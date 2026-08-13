# src/data/dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset
from src.data.indexing import build_pairs
from src.data.preprocessing import nan_to_zero, resize_nearest, remap_mask_values
from src.utils.raster_io import read_raster
from src.utils.normalization import norm_s1_values  
from src.data.legend import load_legend, Legend
from src.data.patching import patch_tensors
from typing import Tuple
import json
import random
import os

class S1Dataset(Dataset):
    def __init__(
        self,
        root_s1: str,
        root_gt: str,
        root_labels: str,
        t_len: int,
        legend: Legend,
        out_size: Tuple[int, int] = (512, 512),
        polarizations: list = ("VH",),
        return_y_long: bool = False,
        patch_size: Tuple[int, int] = None,
        stride: int = None,
        tile_split_path: str = None,
        split: str = "train",
        augmentation_cfg: dict = None,
        pol_map: dict = None,
    ):
        self.pairs = build_pairs(root_s1, root_gt, root_labels, t_len)
        self.t_len = t_len
        self.legend = legend
        self.out_size = out_size
        self.polarizations = list(polarizations)

        self.pol_map = pol_map or {"VH": 0, "VV": 1}

        missing_pols = [pol for pol in self.polarizations if pol not in self.pol_map]
        if missing_pols:
            raise ValueError(f"Polarizations {missing_pols} not found in pol_map keys {list(self.pol_map.keys())}.")

        for pol, idx in self.pol_map.items():
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(f"Invalid band index for polarization '{pol}': {idx}. Must be int >= 0.")

        self.return_y_long = return_y_long
        self.patch_size = patch_size
        self.stride = stride

        if tile_split_path is not None:
            with open(tile_split_path, 'r') as f:
                splits = json.load(f)
            if split == "train":
                selected_tiles = set(splits.get("train_tiles", []))
            elif split == "val":
                selected_tiles = set(splits.get("val_tiles", []))
            else:
                selected_tiles = set()
            filtered_pairs = []
            for gt_path, s1_paths in self.pairs:
                tile_id = self._tile_id_from_gt_path(gt_path)
                if tile_id in selected_tiles:
                    filtered_pairs.append((gt_path, s1_paths))

            self.pairs = filtered_pairs

            if len(self.pairs) == 0:
                raise RuntimeError(
                    f"[S1Dataset] Empty dataset after applying split='{split}'. "
                    f"Check tile_split.json and label JSON consistency."
                )

        self.augmentation_cfg = augmentation_cfg or {}
        self.use_aug = self.augmentation_cfg.get("enabled", False)
        self.p_hflip = 0.5 if self.augmentation_cfg.get("horizontal_flip", False) else 0.0
        self.p_vflip = 0.5 if self.augmentation_cfg.get("vertical_flip", False) else 0.0

    def _tile_id_from_gt_path(self, gt_path: str) -> str:
        """
        Convert GT filename to canonical tile_id.
        Example: /path/20KPF_GT_12_02.tif -> 20KPF_12_02
        """
        name = os.path.splitext(os.path.basename(gt_path))[0]
        return name.replace("_GT_", "_", 1)

    def __len__(self):
        return len(self.pairs)

    def _read_gt(self, path: str) -> np.ndarray:
        gt, _ = read_raster(path)         # (C,H,W) o (H,W) a seconda della tua read_raster
        if gt.ndim == 3:
            gt = gt[0]                    # se è (1,H,W) prendiamo la banda 0
        gt = resize_nearest(gt, self.out_size)
        gt = nan_to_zero(gt)
        gt = remap_mask_values(gt, self.legend.raw_to_class)
        return gt

    def _read_s1_series(self, paths) -> np.ndarray:
        series = []

        for p in paths:
            s1, _ = read_raster(p)  # (C,H,W) or (H,W)

            if s1.ndim == 2:
                bands = [s1]
            else:
                bands = []
                for pol in self.polarizations:
                    band_idx = self.pol_map[pol]
                    bands.append(s1[band_idx])

            bands = np.stack(bands, axis=0)  # (P,H,W)
            bands = norm_s1_values(bands)
            bands = resize_nearest(bands, self.out_size)
            bands = nan_to_zero(bands)

            series.append(bands)

        series = np.stack(series, axis=0)   # (T,P,H,W)
        T, P, H, W = series.shape
        return series.reshape(T * P, H, W)  # (C,H,W)

    def _apply_augmentations(self, x: torch.Tensor, y: torch.Tensor):
        # x: (T,H,W), y: (H,W) or (1,H,W)
        if not self.use_aug:
            return x, y

        if random.random() < self.p_hflip:
            x = torch.flip(x, dims=[-1])
            y = torch.flip(y, dims=[-1])

        if random.random() < self.p_vflip:
            x = torch.flip(x, dims=[-2])
            y = torch.flip(y, dims=[-2])

        return x, y

    def __getitem__(self, idx):
        gt_path, s1_paths = self.pairs[idx]

        x = self._read_s1_series(s1_paths)           # (T,H,W)
        y = self._read_gt(gt_path)                   # (H,W)

        x_t = torch.from_numpy(x).to(torch.float32)  # (T,H,W)

        if self.return_y_long:
            y_t = torch.from_numpy(y).to(torch.long)         # (H,W)
        else:
            y_t = torch.from_numpy(y).to(torch.float32).unsqueeze(0)  # (1,H,W)

        x_t, y_t = self._apply_augmentations(x_t, y_t)

        if self.patch_size is not None and self.stride is not None:
            # Add batch dimension
            x_t = x_t.unsqueeze(0)  # (1, T, H, W)
            if self.return_y_long:
                y_t = y_t.unsqueeze(0)  # (1, H, W)
            else:
                y_t = y_t.unsqueeze(0)  # (1, 1, H, W)

            # Patch x_t: input shape (B, T, H, W)
            x_patched = patch_tensors(x_t, self.patch_size, self.stride)  # (B*P, T, ph, pw)

            # Patch y_t: input shape (B, C, H, W) or (B, H, W)
            y_patched = patch_tensors(y_t, self.patch_size, self.stride)  # (B*P, C, ph, pw) or (B*P, ph, pw)

            return x_patched, y_patched

        return x_t, y_t

if __name__ == "__main__":
    from src.utils.plotting import plot_samples
    from src.utils.config import load_config

    import os

    # ---- LOAD CONFIGS ----
    dataset_cfg = load_config("configs/dataset.yaml")
    legend = load_legend("configs/legend.yaml")

    # ---- DATASET ----
    base_path = dataset_cfg["dataset"]["base_path"]
    region = dataset_cfg["dataset"]["regions"][2]

    region_root = os.path.join(base_path, region)
    root_s1 = os.path.join(region_root, "s1")
    root_gt = os.path.join(region_root, "ground_reference")
    root_labels = os.path.join(region_root, "labels")

    dataset = S1Dataset(
        root_s1=root_s1,
        root_gt=root_gt,
        root_labels=root_labels,
        t_len=dataset_cfg["temporal"]["t_len"],
        legend=legend,
        out_size=tuple(dataset_cfg["spatial"]["out_size"]),
        polarizations=dataset_cfg["spatial"]["polarizations"],
        return_y_long=dataset_cfg["spatial"]["return_y_long"],
        patch_size=tuple(dataset_cfg["patching"]["patch_size"])
            if dataset_cfg["patching"]["enabled"] else None,
        stride=dataset_cfg["patching"]["stride"]
            if dataset_cfg["patching"]["enabled"] else None,
        tile_split_path= f'configs/splits/{region}/tile_split.json',
        split = 'train',
        augmentation_cfg=dataset_cfg.get("augmentation", {}),
    )

    print(f"\nScene-level dataset length: {len(dataset)}")

    # ---- SINGLE SAMPLE CHECK ----
    x, y = dataset[0]
    print("\nSingle scene sample:")
    print("  X shape:", x.shape, x.dtype)
    print("  Y shape:", y.shape, y.dtype)

    # ---- VISUAL SANITY CHECK (RANDOM SAMPLE SUBSET) ----
    print("\nPlotting random samples...")
    plot_samples(
        dataset,
        num_samples=dataset_cfg.get("sanity_check", {}).get("num_samples", 10),
        seed=dataset_cfg.get("sanity_check", {}).get("seed", 42),
    )