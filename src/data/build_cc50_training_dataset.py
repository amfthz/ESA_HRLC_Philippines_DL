from pathlib import Path
import shutil
import csv
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling


ORIGINAL_REGION_ROOT = Path(
    "/home/tlcrs/Philippines_Project/09_Luigi Russo/"
    "DATASET_PHILIPPINES/training/philippines"
)

CC50_FULL_TILE_DIR = Path(
    "/media/tlcrs/Disc_Data/Amin_data/molca_rare_classes/"
    "06_MOLCA_Output_v2_shrub_rescue_conservative_cc50"
)

OUT_REGION_ROOT = Path(
    "/home/tlcrs/Philippines_Project/09_Luigi Russo/"
    "DATASET_PHILIPPINES/training_cc50/philippines"
)

ORIGINAL_GT_DIR = ORIGINAL_REGION_ROOT / "ground_reference"
ORIGINAL_S1_DIR = ORIGINAL_REGION_ROOT / "s1"
ORIGINAL_LABELS_DIR = ORIGINAL_REGION_ROOT / "labels"

OUT_GT_DIR = OUT_REGION_ROOT / "ground_reference"
OUT_S1_LINK = OUT_REGION_ROOT / "s1"
OUT_LABELS_LINK = OUT_REGION_ROOT / "labels"

OUT_REPORT_DIR = OUT_REGION_ROOT / "_cc50_build_report"
OUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = {
    1: "Built-up",
    2: "Cropland",
    3: "Forest",
    4: "Grassland",
    5: "Shrubland",
    6: "Wetland",
    7: "Water",
    8: "Bareland",
    9: "Mangrove",
    255: "Unclassified_255",
}


def scene_id_from_gt_path(path: Path) -> str:
    return path.stem.replace("_GT_", "_", 1)


def base_tile_from_scene_id(scene_id: str) -> str:
    return scene_id.split("_")[0]


def count_values(arr):
    vals, cnts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, cnts)}


def make_symlink(link_path: Path, target_path: Path):
    if link_path.exists() or link_path.is_symlink():
        if link_path.is_symlink():
            current_target = Path(link_path.resolve())
            if current_target == target_path:
                print(f"Symlink already correct: {link_path} -> {target_path}")
                return
            else:
                raise RuntimeError(
                    f"Existing symlink points somewhere else:\n"
                    f"{link_path} -> {current_target}\n"
                    f"Expected: {target_path}"
                )
        else:
            raise RuntimeError(f"Path already exists and is not a symlink: {link_path}")

    link_path.symlink_to(target_path, target_is_directory=True)
    print(f"Created symlink: {link_path} -> {target_path}")


def main():
    print("===== BUILD CC50 TRAINING DATASET =====")
    print("Original region root:", ORIGINAL_REGION_ROOT)
    print("CC50 full-tile dir:", CC50_FULL_TILE_DIR)
    print("Output region root:", OUT_REGION_ROOT)

    for p in [ORIGINAL_GT_DIR, ORIGINAL_S1_DIR, ORIGINAL_LABELS_DIR, CC50_FULL_TILE_DIR]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required path: {p}")

    OUT_REGION_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_GT_DIR.mkdir(parents=True, exist_ok=True)

    make_symlink(OUT_S1_LINK, ORIGINAL_S1_DIR)
    make_symlink(OUT_LABELS_LINK, ORIGINAL_LABELS_DIR)

    gt_files = sorted(ORIGINAL_GT_DIR.glob("*.tif"))
    print("\nOriginal GT scene files:", len(gt_files))

    if len(gt_files) == 0:
        raise RuntimeError("No GT files found.")

    rows = []
    overall_counts = defaultdict(int)
    missing_cc50 = []

    for idx, template_gt in enumerate(gt_files, start=1):
        scene_id = scene_id_from_gt_path(template_gt)
        base_tile = base_tile_from_scene_id(scene_id)

        cc50_tile = (
            CC50_FULL_TILE_DIR / base_tile /
            f"MOLCA_V2_SHRUB_RESCUE_CC50_{base_tile}.tif"
        )

        if not cc50_tile.exists():
            missing_cc50.append((scene_id, str(cc50_tile)))
            continue

        out_gt = OUT_GT_DIR / template_gt.name

        with rasterio.open(template_gt) as ref:
            profile = ref.profile.copy()
            ref_crs = ref.crs
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height

        with rasterio.open(cc50_tile) as src:
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

        arr = arr.astype(np.uint8)

        profile.update(
            dtype="uint8",
            count=1,
            nodata=255,
            compress="lzw",
            tiled=True,
            BIGTIFF="IF_SAFER",
        )

        with rasterio.open(out_gt, "w", **profile) as dst:
            dst.write(arr, 1)

        counts = count_values(arr)
        for k, v in counts.items():
            overall_counts[k] += v

        rows.append({
            "scene_id": scene_id,
            "base_tile": base_tile,
            "output_gt": str(out_gt),
            "total_pixels": int(arr.size),
            "shrubland_pixels_raw": counts.get(5, 0),
            "unclassified_255_pixels_raw": counts.get(255, 0),
            "has_shrubland": counts.get(5, 0) > 0,
        })

        if idx % 25 == 0:
            print(f"Processed {idx}/{len(gt_files)} scenes...")

    if missing_cc50:
        print("\nERROR: Missing CC50 tiles:")
        for item in missing_cc50[:20]:
            print(item)
        raise RuntimeError(f"Missing {len(missing_cc50)} CC50 full-tile rasters.")

    report_csv = OUT_REPORT_DIR / "cc50_training_dataset_scene_summary.csv"

    with open(report_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("\n===== DONE =====")
    print("CC50 ground_reference files written:", len(list(OUT_GT_DIR.glob('*.tif'))))
    print("Report saved:", report_csv)

    print("\n===== OVERALL RAW CC50 LABEL VALUES IN 277 SCENES =====")
    total = sum(overall_counts.values())
    for val in sorted(overall_counts):
        name = CLASS_NAMES.get(val, f"Other_{val}")
        count = overall_counts[val]
        print(f"{val:>3} {name:<20} {count:>12,}  {count/total*100:8.4f}%")

    print("\nOutput dataset:")
    print(OUT_REGION_ROOT)


if __name__ == "__main__":
    main()
