from pathlib import Path
from collections import defaultdict
import json
import csv

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

from src.data.preprocessing import nan_to_zero, resize_nearest, remap_mask_values
from src.data.legend import load_legend
from src.data.patching import patch_tensors
import torch


# --------------------------------------------------
# Paths
# --------------------------------------------------
PIPELINE_ROOT = Path("/home/tlcrs/Philippines_Project/09_Luigi Russo/03-Swin_Unet_Pipeline")

TRAINING_DATASET_ROOT = Path("/home/tlcrs/Philippines_Project/09_Luigi Russo/DATASET_PHILIPPINES")
GT_TEMPLATE_DIR = TRAINING_DATASET_ROOT / "training/philippines/ground_reference"

CC50_FULL_TILE_DIR = Path(
    "/media/tlcrs/Disc_Data/Amin_data/molca_rare_classes/"
    "06_MOLCA_Output_v2_shrub_rescue_conservative_cc50"
)

SPLIT_JSON = PIPELINE_ROOT / "configs/splits/philippines/tile_split.json"
LEGEND_YAML = PIPELINE_ROOT / "configs/legend.yaml"

OUT_DIR = PIPELINE_ROOT / "outputs/cc50_pretraining_checks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SIZE = (512, 512)
PATCH_SIZE = (256, 256)
STRIDE = 128
SHRUB_CLASS = 5


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def scene_id_from_gt_path(path: Path) -> str:
    # Example: 51PVN_GT_16_01.tif -> 51PVN_16_01
    return path.stem.replace("_GT_", "_", 1)


def base_tile_from_scene_id(scene_id: str) -> str:
    # Example: 51PVN_16_01 -> 51PVN
    return scene_id.split("_")[0]


def read_cc50_for_gt_template(cc50_path: Path, gt_template_path: Path) -> np.ndarray:
    """
    Read the CC50 full-tile raster on the exact grid of the existing
    training ground-reference scene.
    """
    with rasterio.open(gt_template_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    with rasterio.open(cc50_path) as src:
        with WarpedVRT(
            src,
            crs=ref_crs,
            transform=ref_transform,
            width=ref_width,
            height=ref_height,
            resampling=Resampling.nearest,
            nodata=255,
        ) as vrt:
            arr = vrt.read(1, masked=False)

    return arr


def loader_like_remap(arr: np.ndarray, raw_to_class: dict) -> np.ndarray:
    """
    Mimics S1Dataset._read_gt:
    resize_nearest -> nan_to_zero -> remap_mask_values.
    Important: 255 is not in legend.yaml, so remap_mask_values turns it into 0.
    """
    arr = resize_nearest(arr, OUT_SIZE)
    arr = nan_to_zero(arr)
    arr = remap_mask_values(arr, raw_to_class)
    return arr


def make_label_patches(y_remapped: np.ndarray) -> np.ndarray:
    """
    Mimics dataset patching for labels:
    y -> tensor float [1,H,W] -> add batch -> patch_tensors -> [9,1,256,256]
    """
    y_t = torch.from_numpy(y_remapped).to(torch.float32).unsqueeze(0)  # [1,H,W]
    y_t = y_t.unsqueeze(0)  # [1,1,H,W]
    patches = patch_tensors(y_t, PATCH_SIZE, STRIDE)  # [N,1,256,256]
    return patches.numpy().astype(np.int64)


def pct(a, b):
    return (a / b * 100) if b else 0.0


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    print("===== CC50 SHRUBLAND TRAIN/VAL + 255 HANDLING CHECK =====")
    print("GT template dir:", GT_TEMPLATE_DIR)
    print("CC50 full-tile dir:", CC50_FULL_TILE_DIR)
    print("Split JSON:", SPLIT_JSON)
    print("Legend YAML:", LEGEND_YAML)
    print("Output dir:", OUT_DIR)

    if not GT_TEMPLATE_DIR.exists():
        raise FileNotFoundError(f"GT template folder not found: {GT_TEMPLATE_DIR}")

    if not CC50_FULL_TILE_DIR.exists():
        raise FileNotFoundError(f"CC50 folder not found: {CC50_FULL_TILE_DIR}")

    with open(SPLIT_JSON, "r") as f:
        split = json.load(f)

    train_ids = set(split["train_tiles"])
    val_ids = set(split["val_tiles"])
    all_split_ids = train_ids | val_ids

    legend = load_legend(str(LEGEND_YAML))
    raw_to_class = legend.raw_to_class

    gt_files = sorted(GT_TEMPLATE_DIR.glob("*.tif"))

    print("\nGT template files found:", len(gt_files))
    print("Train scene IDs:", len(train_ids))
    print("Val scene IDs:", len(val_ids))

    scene_rows = []
    patch_rows = []

    summary = {
        "train": defaultdict(int),
        "val": defaultdict(int),
    }

    tile_summary = {
        "train": defaultdict(lambda: defaultdict(int)),
        "val": defaultdict(lambda: defaultdict(int)),
    }

    unique_before = defaultdict(int)
    unique_after = defaultdict(int)

    missing_cc50 = []
    missing_from_split = []
    processed_scenes = 0

    for gt_path in gt_files:
        scene_id = scene_id_from_gt_path(gt_path)

        if scene_id in train_ids:
            split_name = "train"
        elif scene_id in val_ids:
            split_name = "val"
        else:
            missing_from_split.append(scene_id)
            continue

        base_tile = base_tile_from_scene_id(scene_id)
        cc50_path = (
            CC50_FULL_TILE_DIR / base_tile /
            f"MOLCA_V2_SHRUB_RESCUE_CC50_{base_tile}.tif"
        )

        if not cc50_path.exists():
            missing_cc50.append((scene_id, str(cc50_path)))
            continue

        # Read CC50 label on this scene grid
        raw_arr = read_cc50_for_gt_template(cc50_path, gt_path)

        # Count raw values before loader-like remapping
        vals, cnts = np.unique(raw_arr, return_counts=True)
        for v, c in zip(vals, cnts):
            unique_before[int(v)] += int(c)

        raw_255_pixels = int(np.sum(raw_arr == 255))
        raw_shrub_pixels = int(np.sum(raw_arr == SHRUB_CLASS))

        # Mimic loader remap/resizing
        remapped = loader_like_remap(raw_arr, raw_to_class)

        vals, cnts = np.unique(remapped, return_counts=True)
        for v, c in zip(vals, cnts):
            unique_after[int(v)] += int(c)

        after_255_pixels = int(np.sum(remapped == 255))
        after_zero_pixels = int(np.sum(remapped == 0))
        after_shrub_pixels = int(np.sum(remapped == SHRUB_CLASS))

        # Make the actual 256x256 patches used in training
        y_patches = make_label_patches(remapped)

        n_patches = int(y_patches.shape[0])
        patch_shrub_pixels = []
        patch_valid_pixels = []
        patch_zero_pixels = []

        for patch_idx in range(n_patches):
            p = y_patches[patch_idx, 0]
            shrub_count = int(np.sum(p == SHRUB_CLASS))
            zero_count = int(np.sum(p == 0))
            valid_count = int(p.size - zero_count)

            patch_shrub_pixels.append(shrub_count)
            patch_valid_pixels.append(valid_count)
            patch_zero_pixels.append(zero_count)

            patch_rows.append({
                "split": split_name,
                "scene_id": scene_id,
                "base_tile": base_tile,
                "patch_idx": patch_idx,
                "shrub_pixels": shrub_count,
                "valid_pixels": valid_count,
                "zero_ignored_pixels": zero_count,
                "has_shrubland": shrub_count > 0,
                "has_shrubland_ge_10px": shrub_count >= 10,
                "has_shrubland_ge_50px": shrub_count >= 50,
                "has_shrubland_ge_100px": shrub_count >= 100,
                "has_shrubland_ge_500px": shrub_count >= 500,
                "has_shrubland_ge_1000px": shrub_count >= 1000,
            })

        scene_has_shrub = after_shrub_pixels > 0
        scene_has_raw_255 = raw_255_pixels > 0

        scene_rows.append({
            "split": split_name,
            "scene_id": scene_id,
            "base_tile": base_tile,
            "raw_255_pixels_before_loader": raw_255_pixels,
            "raw_shrub_pixels_before_loader": raw_shrub_pixels,
            "remapped_255_pixels_after_loader": after_255_pixels,
            "remapped_zero_pixels_after_loader": after_zero_pixels,
            "remapped_shrub_pixels_after_loader": after_shrub_pixels,
            "scene_has_shrubland_after_loader": scene_has_shrub,
            "scene_has_raw_255_before_loader": scene_has_raw_255,
            "n_patches": n_patches,
            "patches_with_any_shrubland": int(sum(x > 0 for x in patch_shrub_pixels)),
            "patches_with_shrubland_ge_10px": int(sum(x >= 10 for x in patch_shrub_pixels)),
            "patches_with_shrubland_ge_50px": int(sum(x >= 50 for x in patch_shrub_pixels)),
            "patches_with_shrubland_ge_100px": int(sum(x >= 100 for x in patch_shrub_pixels)),
            "patches_with_shrubland_ge_500px": int(sum(x >= 500 for x in patch_shrub_pixels)),
            "patches_with_shrubland_ge_1000px": int(sum(x >= 1000 for x in patch_shrub_pixels)),
        })

        # Split summary
        s = summary[split_name]
        s["scenes"] += 1
        s["scenes_with_shrubland"] += int(scene_has_shrub)
        s["scenes_with_raw_255"] += int(scene_has_raw_255)
        s["raw_255_pixels_before_loader"] += raw_255_pixels
        s["remapped_255_pixels_after_loader"] += after_255_pixels
        s["remapped_zero_pixels_after_loader"] += after_zero_pixels
        s["shrub_pixels_after_loader"] += after_shrub_pixels
        s["patches"] += n_patches
        s["patches_with_any_shrubland"] += int(sum(x > 0 for x in patch_shrub_pixels))
        s["patches_with_shrubland_ge_10px"] += int(sum(x >= 10 for x in patch_shrub_pixels))
        s["patches_with_shrubland_ge_50px"] += int(sum(x >= 50 for x in patch_shrub_pixels))
        s["patches_with_shrubland_ge_100px"] += int(sum(x >= 100 for x in patch_shrub_pixels))
        s["patches_with_shrubland_ge_500px"] += int(sum(x >= 500 for x in patch_shrub_pixels))
        s["patches_with_shrubland_ge_1000px"] += int(sum(x >= 1000 for x in patch_shrub_pixels))

        # Tile summary
        ts = tile_summary[split_name][base_tile]
        ts["scenes"] += 1
        ts["scenes_with_shrubland"] += int(scene_has_shrub)
        ts["shrub_pixels_after_loader"] += after_shrub_pixels
        ts["patches"] += n_patches
        ts["patches_with_any_shrubland"] += int(sum(x > 0 for x in patch_shrub_pixels))
        ts["patches_with_shrubland_ge_50px"] += int(sum(x >= 50 for x in patch_shrub_pixels))
        ts["patches_with_shrubland_ge_100px"] += int(sum(x >= 100 for x in patch_shrub_pixels))

        processed_scenes += 1

    # Save CSVs
    scene_csv = OUT_DIR / "cc50_scene_level_shrubland_train_val.csv"
    patch_csv = OUT_DIR / "cc50_patch_level_shrubland_train_val.csv"
    tile_csv = OUT_DIR / "cc50_tile_level_shrubland_distribution_train_val.csv"
    unique_csv = OUT_DIR / "cc50_255_loader_remap_unique_values.csv"

    with open(scene_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scene_rows[0].keys()))
        writer.writeheader()
        writer.writerows(scene_rows)

    with open(patch_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(patch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(patch_rows)

    tile_rows = []
    for split_name in ["train", "val"]:
        for base_tile, vals in tile_summary[split_name].items():
            tile_rows.append({
                "split": split_name,
                "base_tile": base_tile,
                **dict(vals),
            })

    with open(tile_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(tile_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tile_rows)

    unique_rows = []
    for v in sorted(set(unique_before.keys()) | set(unique_after.keys())):
        unique_rows.append({
            "value": v,
            "pixels_before_loader_remap": unique_before.get(v, 0),
            "pixels_after_loader_remap": unique_after.get(v, 0),
        })

    with open(unique_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(unique_rows[0].keys()))
        writer.writeheader()
        writer.writerows(unique_rows)

    # Print summary
    print("\n===== FILES SAVED =====")
    print(scene_csv)
    print(patch_csv)
    print(tile_csv)
    print(unique_csv)

    print("\n===== BASIC CHECK =====")
    print("Processed scenes:", processed_scenes)
    print("Missing from split:", len(missing_from_split))
    print("Missing CC50 full-tile rasters:", len(missing_cc50))

    if missing_cc50:
        print("\nFirst missing CC50 examples:")
        for x in missing_cc50[:10]:
            print(x)

    print("\n===== LUIGI CHECK A: CC50 SHRUBLAND IN TRAIN / VAL =====")
    for split_name in ["train", "val"]:
        s = summary[split_name]
        print(f"\n{split_name.upper()}:")
        print(f"  scenes: {s['scenes']:,}")
        print(
            f"  scenes with Shrubland: {s['scenes_with_shrubland']:,} "
            f"({pct(s['scenes_with_shrubland'], s['scenes']):.2f}%)"
        )
        print(f"  total 256x256 patches: {s['patches']:,}")
        print(
            f"  patches with any Shrubland: {s['patches_with_any_shrubland']:,} "
            f"({pct(s['patches_with_any_shrubland'], s['patches']):.2f}%)"
        )
        print(
            f"  patches with Shrubland >=10 px: {s['patches_with_shrubland_ge_10px']:,} "
            f"({pct(s['patches_with_shrubland_ge_10px'], s['patches']):.2f}%)"
        )
        print(
            f"  patches with Shrubland >=50 px: {s['patches_with_shrubland_ge_50px']:,} "
            f"({pct(s['patches_with_shrubland_ge_50px'], s['patches']):.2f}%)"
        )
        print(
            f"  patches with Shrubland >=100 px: {s['patches_with_shrubland_ge_100px']:,} "
            f"({pct(s['patches_with_shrubland_ge_100px'], s['patches']):.2f}%)"
        )
        print(
            f"  patches with Shrubland >=500 px: {s['patches_with_shrubland_ge_500px']:,} "
            f"({pct(s['patches_with_shrubland_ge_500px'], s['patches']):.2f}%)"
        )
        print(
            f"  patches with Shrubland >=1000 px: {s['patches_with_shrubland_ge_1000px']:,} "
            f"({pct(s['patches_with_shrubland_ge_1000px'], s['patches']):.2f}%)"
        )
        print(f"  total Shrubland pixels after loader/remap: {s['shrub_pixels_after_loader']:,}")

    print("\n===== TOP TRAIN TILES BY SHRUBLAND PATCHES ANY =====")
    train_tiles_sorted = sorted(
        tile_summary["train"].items(),
        key=lambda kv: kv[1]["patches_with_any_shrubland"],
        reverse=True,
    )
    for base_tile, vals in train_tiles_sorted[:20]:
        print(
            f"{base_tile}: scenes={vals['scenes']:,} | "
            f"scenes_with_shrub={vals['scenes_with_shrubland']:,} | "
            f"patches_with_shrub={vals['patches_with_any_shrubland']:,} | "
            f"patches_ge50={vals['patches_with_shrubland_ge_50px']:,} | "
            f"shrub_pixels={vals['shrub_pixels_after_loader']:,}"
        )

    print("\n===== TOP VAL TILES BY SHRUBLAND PATCHES ANY =====")
    val_tiles_sorted = sorted(
        tile_summary["val"].items(),
        key=lambda kv: kv[1]["patches_with_any_shrubland"],
        reverse=True,
    )
    for base_tile, vals in val_tiles_sorted[:20]:
        print(
            f"{base_tile}: scenes={vals['scenes']:,} | "
            f"scenes_with_shrub={vals['scenes_with_shrubland']:,} | "
            f"patches_with_shrub={vals['patches_with_any_shrubland']:,} | "
            f"patches_ge50={vals['patches_with_shrubland_ge_50px']:,} | "
            f"shrub_pixels={vals['shrub_pixels_after_loader']:,}"
        )

    print("\n===== LUIGI CHECK B: 255 HANDLING =====")
    for split_name in ["train", "val"]:
        s = summary[split_name]
        print(f"\n{split_name.upper()}:")
        print(f"  scenes with raw 255 before loader/remap: {s['scenes_with_raw_255']:,}")
        print(f"  raw 255 pixels before loader/remap: {s['raw_255_pixels_before_loader']:,}")
        print(f"  255 pixels after loader/remap: {s['remapped_255_pixels_after_loader']:,}")
        print(f"  zero/ignored pixels after loader/remap: {s['remapped_zero_pixels_after_loader']:,}")

    print("\nUnique values before loader/remap:")
    for v in sorted(unique_before):
        print(f"  {v}: {unique_before[v]:,}")

    print("\nUnique values after loader/remap:")
    for v in sorted(unique_after):
        print(f"  {v}: {unique_after[v]:,}")

    if unique_after.get(255, 0) == 0:
        print("\nRESULT: 255 does not enter the loss. It is remapped to 0 and ignored.")
    else:
        print("\nWARNING: 255 remains after loader remap. Training would need fixing before proceeding.")


if __name__ == "__main__":
    main()
