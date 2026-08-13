#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
check_tiles_on_s3.py

Check which tiles have complete data available on AWS S3:
- 24 seasonal SAR features
- 4 super-images
- water product (seasonality and/or no-seasonality)

This script DOES NOT download anything.
It only verifies existence via AWS CLI.
"""

import argparse
import subprocess
import sys


# --------------------------------------------------
# CLI
# --------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Check tile completeness on AWS S3"
    )

    p.add_argument("--bucket", required=True, help="S3 bucket name")
    p.add_argument("--year", required=True, type=int)
    p.add_argument("--area", required=True, help="Area name")
    p.add_argument("--source", required=True, help="Source name")

    p.add_argument(
        "--check-seasonality",
        action="store_true",
        help="Check seasonal water product"
    )
    p.add_argument(
        "--check-no-seasonality",
        action="store_true",
        help="Check non-seasonal water product"
    )

    return p.parse_args()


# --------------------------------------------------
# AWS helper
# --------------------------------------------------
def s3_ls(path: str):
    cmd = ["aws", "s3", "ls", path]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # If path does not exist, return empty list
    if result.returncode != 0:
        return []

    return result.stdout.strip().splitlines()


def list_tiles(bucket, year):
    base = f"s3://{bucket}/data/preprocessed/sar/seasonal_features/{year}/"
    lines = s3_ls(base)

    tiles = []
    for l in lines:
        parts = l.strip().split()
        if len(parts) >= 2 and parts[-1].endswith("/"):
            tiles.append(parts[-1].rstrip("/"))

    return sorted(tiles)


# --------------------------------------------------
# Checks
# --------------------------------------------------
def check_seasonal_features(bucket, year, tile):
    base = f"s3://{bucket}/data/preprocessed/sar/seasonal_features/{year}/{tile}/vh/"
    lines = s3_ls(base)

    seasonal_files = [
        l.split()[-1]
        for l in lines
        if l.strip().endswith(".tif")
        and not l.strip().endswith("_registered.tif")
        and not l.strip().endswith("_VARIANCE.tif")
    ]

    count = len(seasonal_files)

    print(f"  Seasonal spatial features found: {count}")
    return count == 24


def check_super_images(bucket, year, tile):
    # --- First try seasonal_features ---
    base_seasonal = f"s3://{bucket}/data/preprocessed/sar/seasonal_features/{year}/{tile}/vh/"
    lines_seasonal = s3_ls(base_seasonal)

    seasonal_super = [
        l.split()[-1]
        for l in lines_seasonal
        if l.strip().endswith("_registered.tif")
    ]

    if len(seasonal_super) == 4:
        print("  Super-images found (seasonal_features): 4")
        return True

    # --- Fallback to coreg ---
    base_coreg = f"s3://{bucket}/data/preprocessed/sar/coreg/{year}/{tile}/vh/"
    lines_coreg = s3_ls(base_coreg)

    coreg_super = [
        l.split()[-1]
        for l in lines_coreg
        if l.strip().endswith("_registered.tif")
        and any(season in l for season in ["autumn", "spring", "summer", "winter"])
    ]

    if len(coreg_super) == 4:
        print("  Super-images found (fallback coreg): 4")
        return True

    print(
        f"  Super-images found: seasonal={len(seasonal_super)}, "
        f"coreg={len(coreg_super)}"
    )
    return False


def check_water(bucket, year, area, tile, mode):
    area_lower = area.lower()
    base = f"s3://{bucket}/data/preprocessed/sar/water/{year}/{area_lower}/{tile}/"
    lines = s3_ls(base)

    s3_files = [l.split()[-1] for l in lines if l.endswith(".tif")]

    if mode == "seasonality":
        expected = f"{tile}_water_seasonality_masked.tif"
    else:
        expected = f"{tile}_water_no_seasonality_masked.tif"

    found = expected in s3_files

    print(f"  Water ({mode}) map found: {int(found)}")
    return found


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    args = parse_args()

    print("\n[CHECK] Verifying tile on AWS")
    print(f"  Bucket : {args.bucket}")
    print(f"  Year   : {args.year}")
    print(f"  Area   : {args.area}")
    print("")

    print("[CHECK] Listing tiles available for this year...")
    tiles = list_tiles(args.bucket, args.year)

    if not tiles:
        print("No tiles found.")
        sys.exit(0)

    complete_tiles = []

    for tile in tiles:
        print(f"\nChecking tile: {tile}")

        ok_features = check_seasonal_features(args.bucket, args.year, tile)
        ok_super = check_super_images(args.bucket, args.year, tile)

        ok_water_seas = True
        ok_water_no = True

        if args.check_seasonality:
            ok_water_seas = check_water(
                args.bucket,
                args.year,
                args.area,
                tile,
                mode="seasonality"
            )

        if args.check_no_seasonality:
            ok_water_no = check_water(
                args.bucket,
                args.year,
                args.area,
                tile,
                mode="no_seasonality"
            )

        if ok_features and ok_super and ok_water_seas and ok_water_no:
            complete_tiles.append(tile)

    print("\n======================================")
    print("COMPLETE TILES:")
    for t in complete_tiles:
        print(t)

    print(f"\nTotal complete tiles: {len(complete_tiles)}")
    print("======================================\n")


if __name__ == "__main__":
    main()

# To execute, run:
# python -m src.check_tiles_on_s3 \
#   --bucket cci-hrlc-phase-2 \
#   --year 2019 \
#   --area amazon \
#   --source esa \
#   --check-seasonality
#   --check-no-seasonality