#!/usr/bin/env python3

import subprocess
import argparse
import json
from pathlib import Path

import shutil


# ============================================================
# USER CONFIG
# ============================================================

BASE_LOCAL_DIR = Path("/Users/luigi/Downloads/input_features_ESA_CCI")

S3_SEASONAL_BASE = "s3://cci-hrlc-phase-2/data/preprocessed/sar/seasonal_features"
S3_WATER_BASE = "s3://cci-hrlc-phase-2/data/preprocessed/sar/water"

identifiers = {
    "type": "static",
    "years": [2019],
    "area": "Amazon",   # Africa | Amazon | Siberia
    "tiles": ["19LEL"],
    "source": "S1",
}


POLARIZATION = "vh"

# ============================================================
# RENAME CONFIG (FINAL NOMENCLATURE)
# ============================================================

SEASON_CODE = {
    "winter": "01",
    "spring": "02",
    "summer": "03",
    "autumn": "04",
}

FEATURE_CODE = {
    "SI": "01",
    "LEE": "02",
    "MAX": "03",
    "MIN": "04",
    "MAXMIN": "05",
    "MEAN": "06",
    "MEDIAN": "07",
    # VARIANCE intentionally excluded (not part of final 28 features)
}


def extract_feature_from_name(name: str):
    """
    Extract feature name from S3 filename.
    Handles:
    - *_registered.tif  -> SI
    - *_registered_<FEATURE>.tif
    """
    if name.endswith("_registered.tif"):
        return "SI"

    for key in FEATURE_CODE.keys():
        if f"_registered_{key}.tif" in name:
            return key

    return None


def rename_downloaded_features(features_path: Path, year: int, tile: str):
    """
    Rename downloaded seasonal features and superimages
    into final ESA-CCI naming convention:

    {tile}_{year}{season_index}{feature_index}.tif
    """

    for tif in features_path.glob("*.tif"):
        name = tif.name

        parts = name.replace(".tif", "").split("_")

        # Expected structure:
        # 18MYS_autumn_vh_2019_registered_MAX.tif
        if len(parts) < 5:
            continue

        tile_name = parts[0]
        season = parts[1].lower()

        if season not in SEASON_CODE:
            continue

        feature = extract_feature_from_name(name)

        if feature is None:
            continue

        season_code = SEASON_CODE[season]
        feature_code = FEATURE_CODE[feature]

        # Final naming format:
        # {tile}_{year}{season_code}{feature_code}_{FEATURE}.tif
        # Example: 18MYS_20190102_LEE.tif
        new_name = f"{tile_name}_{year}{season_code}{feature_code}_{feature}.tif"
        new_path = features_path / new_name

        # Avoid overwriting silently
        if new_path.exists():
            print(f"[SKIP] {new_name} already exists")
            continue

        print(f"Renaming: {name} -> {new_name}")
        tif.rename(new_path)


# ============================================================
# PATH BUILDERS
# ============================================================

def build_local_path(base_dir: Path, ids: dict, year: int, tile: str):
    return (
        base_dir
        / ids["type"]
        / str(year)
        / ids["area"]
        / tile
        / ids["source"]
    )


def build_s3_seasonal_path(year: int, tile: str):
    return f"{S3_SEASONAL_BASE}/{year}/{tile}/{POLARIZATION}/"


def build_s3_water_path(ids: dict, year: int, tile: str):
    area_lower = ids["area"].lower()
    return f"{S3_WATER_BASE}/{year}/{area_lower}/{tile}/"


# ============================================================
# AWS SYNC
# ============================================================

def aws_sync(s3_path: str, local_path: Path):
    print(f"\n⬇ Syncing from:\n{s3_path}")
    subprocess.run(
        ["aws", "s3", "sync", s3_path, str(local_path)],
        check=True
    )


# ============================================================
# WATER MAP DOWNLOAD AND HANDLING
# ============================================================

def download_and_handle_water(s3_path: str, local_path: Path, tile: str, mode: str):
    """
    Download water map from S3 and handle naming depending on mode.

    NOTE:
    We currently assume that the non-seasonality product on S3 might still be named
    '<tile>_water_seasonality_masked.tif'. This MUST be verified once non-seasonal
    products are available on S3. If S3 uses a different naming convention
    (e.g. '<tile>_water_no_seasonality_masked.tif'), this logic should be updated.
    """

    # Sync all files from water S3 folder
    aws_sync(s3_path, local_path)

    seasonal_name = f"{tile}_water_seasonality_masked.tif"
    no_seasonal_name = f"{tile}_water_no_seasonality_masked.tif"

    seasonal_file = local_path / seasonal_name
    no_seasonal_file = local_path / no_seasonal_name

    if mode == "seasonal":
        if seasonal_file.exists():
            print(f"✔ Seasonal water map found: {seasonal_name}")
        else:
            print(f"⚠ Expected seasonal water map not found for tile {tile}")
    else:
        # For no-seasonality mode:
        # If S3 already provides a no-seasonality file, keep it.
        if no_seasonal_file.exists():
            print(f"✔ No-seasonality water map found directly on S3: {no_seasonal_name}")
        # Otherwise, if only seasonal exists, rename it locally.
        elif seasonal_file.exists():
            shutil.move(seasonal_file, no_seasonal_file)
            print(f"✔ Renamed {seasonal_name} -> {no_seasonal_name}")
        else:
            print(f"⚠ No water map found for tile {tile} in no-seasonality mode")


# ============================================================
# JSON CREATION
# ============================================================

def create_merge_config(path: Path, ids: dict, year: int, tile: str, mode: str):

    if mode == "seasonal":

        expected_name = f"{tile}_water_seasonality_masked.tif"
        tif_files = list(path.glob(expected_name))
        if len(tif_files) != 1:
            raise RuntimeError(
                f"Expected seasonal water mask '{expected_name}' in {path}, "
                f"found {len(tif_files)} matching files."
            )
        image_path = str(tif_files[0].resolve())

        config = {
            "water": {
                "image_path": image_path,
                "has_seasonal": True,
                "seasonal": {
                    "1": 70,
                    "0": 5,
                    "position": 10,
                    "cls_value": 14
                },
                "permanent": {
                    "1": 70,
                    "0": 5,
                    "position": 11,
                    "cls_value": 15
                }
            }
        }

        filename = "merge_config.json"

    else:

        expected_name = f"{tile}_water_no_seasonality_masked.tif"
        tif_files = list(path.glob(expected_name))
        if len(tif_files) != 1:
            raise RuntimeError(
                f"Expected no-seasonality water mask '{expected_name}' in {path}, "
                f"found {len(tif_files)} matching files."
            )
        image_path = str(tif_files[0].resolve())

        config = {
            "water": {
                "image_path": image_path,
                "has_seasonal": True,
                "remove_permanent": True,
                "seasonal": {
                    "1": 70,
                    "0": 5,
                    "position": 10,
                    "cls_value": 15
                },
                "permanent": {
                    "1": 0,
                    "0": 0,
                    "position": 11,
                    "cls_value": 15
                }
            }
        }

        filename = "merge_config_no_seasonality.json"

    path.mkdir(parents=True, exist_ok=True)

    with open(path / filename, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✔ Created {filename} for tile {tile}")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seasonality_input", action="store_true")
    parser.add_argument("--no_seasonality_input", action="store_true")
    args = parser.parse_args()

    if not args.seasonality_input and not args.no_seasonality_input:
        raise ValueError("Specify at least one of: --seasonality_input or --no_seasonality_input")

    for year in identifiers["years"]:
        for tile in identifiers["tiles"]:

            print("\n===================================================")
            print(f"Processing Year: {year} | Tile: {tile}")
            print("===================================================")

            local_main = build_local_path(BASE_LOCAL_DIR, identifiers, year, tile)
            features_path = local_main / "features"

            # Create base features folder
            features_path.mkdir(parents=True, exist_ok=True)

            # ---- PRE-CHECK ON S3 (DO NOT DOWNLOAD IF INCOMPLETE) ----
            seasonal_s3_path = build_s3_seasonal_path(year, tile)

            print("\n🔎 Checking availability of required inputs on S3...")

            # Count seasonal features (excluding VARIANCE)
            seasonal_ls = subprocess.check_output(
                ["aws", "s3", "ls", seasonal_s3_path],
                text=True
            ).splitlines()

            # Count seasonal features (excluding VARIANCE and excluding pure superimages)
            seasonal_s3_files = [
                line.split()[-1]
                for line in seasonal_ls
                if line.strip().endswith(".tif")
                and not line.strip().endswith("_registered.tif")
                and not line.strip().endswith("_VARIANCE.tif")
            ]

            # Count superimages directly inside seasonal_features
            superimage_s3_files = [
                line.split()[-1]
                for line in seasonal_ls
                if line.strip().endswith("_registered.tif")
            ]

            # ---- Check water map presence on S3 (depending on CLI flags) ----
            water_s3_path = build_s3_water_path(identifiers, year, tile)

            try:
                water_ls = subprocess.check_output(
                    ["aws", "s3", "ls", water_s3_path],
                    text=True
                ).splitlines()
            except subprocess.CalledProcessError:
                print("\n⚠ Water S3 path not found.")
                print("   Skipping this tile and moving to the next one.\n")
                continue

            expected_water_files = []

            if args.seasonality_input:
                expected_water_files.append(
                    f"{tile}_water_seasonality_masked.tif"
                )

            if args.no_seasonality_input:
                expected_water_files.append(
                    f"{tile}_water_no_seasonality_masked.tif"
                )

            water_s3_files = [line.split()[-1] for line in water_ls]

            missing_water = [
                fname for fname in expected_water_files
                if fname not in water_s3_files
            ]

            if (
                len(seasonal_s3_files) != 24
                or len(superimage_s3_files) != 4
                or len(missing_water) > 0
            ):
                print("\n⚠ Input data incomplete on S3.")
                print(f"   Seasonal features found on S3: {len(seasonal_s3_files)} (expected 24)")
                print(f"   Superimages found on S3: {len(superimage_s3_files)} (expected 4)")
                if len(missing_water) > 0:
                    print(f"   Missing water maps on S3: {missing_water}")
                print("   Skipping this tile and moving to the next one.\n")
                continue

            print("✔ All required inputs available on S3. Proceeding with download.\n")

            # ---- SYNC seasonal_features (exclude VARIANCE) ----
            seasonal_s3_path = build_s3_seasonal_path(year, tile)
            print(f"\n⬇ Syncing seasonal features (excluding VARIANCE) from:\n{seasonal_s3_path}")
            subprocess.run(
                [
                    "aws", "s3", "sync",
                    seasonal_s3_path,
                    str(features_path),
                    "--exclude", "*_VARIANCE.tif"
                ],
                check=True
            )

            # ---- RENAME FEATURES TO FINAL ESA-CCI SCHEMA ----
            rename_downloaded_features(features_path, year, tile)

            # ---- WATER MAP DOWNLOAD ----
            # Download water map directly into the standard directory expected by infer.py
            water_s3 = build_s3_water_path(identifiers, year, tile)
            water_map_path = local_main / "water_map"
            water_map_path.mkdir(parents=True, exist_ok=True)

            if args.seasonality_input:
                download_and_handle_water(water_s3, water_map_path, tile, "seasonal")

            if args.no_seasonality_input:
                download_and_handle_water(water_s3, water_map_path, tile, "no_seasonal")

    print("\n✅ All tiles processed successfully.")


if __name__ == "__main__":
    main()

# EXAMPLE LAUNCH:
# conda activate aws
# 1) python src/esa_cci_s3_data_manager_download.py --seasonality_input --no_seasonality_input
# 2) python src/esa_cci_s3_data_manager_download.py --seasonality_input
# 3) python src/esa_cci_s3_data_manager_download.py --seasonality_input --no_seasonality_input