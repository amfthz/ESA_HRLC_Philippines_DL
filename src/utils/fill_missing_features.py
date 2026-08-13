

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
fill_missing_features_and_water_map.py

Utility module for preparing the input directory expected by the
ESA CCI SAR DL inference pipeline.

It performs two tasks:

1) Ensure the features directory contains the full set of 28 SAR features
   expected by the model. Missing rasters are generated using a valid
   reference raster footprint.

2) Ensure a water map exists inside the water_map directory. If no
   water map is found, a zero-valued raster is generated automatically
   using the same footprint as the reference feature.

Expected input structure:

input_dir/
    features/
        *.tif
    water_map/
        *.tif

The generated water map will follow the naming convention:

    <tile>_water_no_seasonality_masked.tif

which is compatible with the downstream merging pipeline.
"""

import rasterio
import numpy as np
from pathlib import Path

EXPECTED_STATS = ["SI", "LEE", "MAX", "MIN", "MAXMIN", "MEAN", "MEDIAN"]
EXPECTED_MONTHS = ["01", "02", "03", "04"]


def extract_year(existing_files):
    parts = existing_files[0].stem.split("_")
    date_block = parts[1]
    return date_block[:4]


def extract_tile(existing_files):
    return existing_files[0].stem.split("_")[0]


def pick_best_reference(existing_files):
    """
    Choose raster with minimum fraction of nodata pixels.
    """

    best = None
    best_valid_frac = -1.0

    for p in existing_files:
        with rasterio.open(p) as src:
            m = src.read_masks(1)
            valid_frac = (m > 0).mean()

        if valid_frac > best_valid_frac:
            best_valid_frac = valid_frac
            best = p

    return best, best_valid_frac


def create_empty_feature(reference_path: Path, output_path: Path):
    """
    Create missing feature raster aligned with tile footprint.

    Pixel rule:
      mask == 0 → outside tile → NaN
      mask == 1 → inside tile → 0
    """

    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
        valid_mask = src.read_masks(1) > 0
        H, W = src.height, src.width

    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress="deflate",
        nodata=np.nan
    )

    out = np.zeros((H, W), dtype=np.float32)
    out[~valid_mask] = np.nan

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out, 1)

    print(f"✓ Created missing feature: {output_path.name}")


def create_empty_water_map(reference_path: Path, water_map_path: Path):
    """
    Create a zero water map if none exists.

    All pixels inside tile = 0 (no water).
    """

    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
        valid_mask = src.read_masks(1) > 0
        H, W = src.height, src.width

    profile.update(
        dtype=rasterio.uint8,
        count=1,
        compress="deflate",
        nodata=0
    )

    water = np.zeros((H, W), dtype=np.uint8)

    water_map_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(water_map_path, "w", **profile) as dst:
        dst.write(water, 1)

    print(f"✓ Created empty water map: {water_map_path.name}")


def fill_missing_features(feature_dir: Path):

    existing_files = sorted(feature_dir.glob("*.tif"))

    if not existing_files:
        raise RuntimeError("No raster found in features directory")

    tile_id = extract_tile(existing_files)
    year = extract_year(existing_files)

    reference_raster, vf = pick_best_reference(existing_files)

    print("\n[FEATURE COMPLETION]")
    print("Tile:", tile_id)
    print("Year:", year)
    print("Reference raster:", reference_raster.name, f"(valid frac={vf:.3f})")

    existing_names = {f.name for f in existing_files}

    expected_names = []

    for month in EXPECTED_MONTHS:
        for idx, stat in enumerate(EXPECTED_STATS, start=1):
            day = f"{idx:02d}"
            expected_names.append(f"{tile_id}_{year}{month}{day}_{stat}.tif")

    missing = []

    for name in expected_names:
        if name not in existing_names:
            missing.append(name)
            create_empty_feature(reference_raster, feature_dir / name)

    print("\nFeature summary")
    print("Existing :", len(existing_names))
    print("Expected :", len(expected_names))
    print("Created  :", len(missing))



def ensure_water_map(input_dir: Path, reference_raster: Path):

    water_dir = input_dir / "water_map"
    water_dir.mkdir(parents=True, exist_ok=True)

    existing_water = list(water_dir.glob("*.tif"))

    if existing_water:
        print("\n[WATER MAP]")
        print("Existing water map found:", existing_water[0].name)
        return

    tile = reference_raster.stem.split("_")[0]

    water_name = f"{tile}_water_no_seasonality_masked.tif"
    water_path = water_dir / water_name

    create_empty_water_map(reference_raster, water_path)



def prepare_input_directory(input_dir: str):

    input_dir = Path(input_dir)

    feature_dir = input_dir / "features"

    if not feature_dir.exists():
        raise RuntimeError(f"Features directory not found: {feature_dir}")

    existing_files = sorted(feature_dir.glob("*.tif"))

    if not existing_files:
        raise RuntimeError("No feature rasters found")

    reference_raster, _ = pick_best_reference(existing_files)

    fill_missing_features(feature_dir)

    ensure_water_map(input_dir, reference_raster)


if __name__ == "__main__":

    prepare_input_directory(
        "/home/silvia/Desktop/GIGI/ESA_CCI_PROJECT/tiles_io/input/static/2024/Amazon/23KPQ/S1"
    )