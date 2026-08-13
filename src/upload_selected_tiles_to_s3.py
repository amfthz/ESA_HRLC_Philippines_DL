#!/usr/bin/env python3

import os
import argparse
import subprocess
from pathlib import Path


# --------------------------------------------------
# CLI
# --------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload selected DL tiles to S3"
    )

    parser.add_argument(
        "--local-root",
        required=True,
        help="Local root directory containing tile folders"
    )

    parser.add_argument(
        "--s3-base-path",
        required=True,
        help="S3 base path where tile folders will be created"
    )

    parser.add_argument(
        "--tiles",
        required=True,
        nargs="+",
        help="List of tile names to upload (space separated)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate upload without actually transferring files to S3"
    )

    return parser.parse_args()


# --------------------------------------------------
# Helper
# --------------------------------------------------

def find_required_files(tile_folder: Path):
    """
    Search recursively inside tile folder for:
    - *_classified_DL.tif
    - *_posteriors_DL.tif
    - *_posteriors_DL.json
    """

    classified = None
    posteriors = None
    json_file = None

    for root, dirs, files in os.walk(tile_folder):
        for f in files:
            if f.endswith("_classified_DL.tif"):
                classified = Path(root) / f
            elif f.endswith("_posteriors_DL.tif"):
                posteriors = Path(root) / f
            elif f.endswith("_posteriors_DL.json"):
                json_file = Path(root) / f

    return classified, posteriors, json_file


def upload_file(local_file: Path, s3_dest: str, dry_run: bool = False):
    cmd = [
        "aws", "s3", "cp",
        str(local_file),
        s3_dest
    ]

    if dry_run:
        print(f"[DRY-RUN] aws s3 cp {local_file} {s3_dest}")
        return

    subprocess.run(cmd, check=True)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    args = parse_args()

    local_root = Path(args.local_root).resolve()
    s3_base = args.s3_base_path.rstrip("/")
    tile_list = args.tiles
    dry_run = args.dry_run

    if not local_root.exists():
        raise RuntimeError(f"Local root does not exist: {local_root}")

    print("\n===========================================")
    print("Selected tiles to upload:")
    for t in tile_list:
        print(f"  - {t}")
    print("===========================================\n")

    for tile_name in tile_list:

        tile_folder = local_root / tile_name

        if not tile_folder.exists():
            print(f"[WARNING] Local folder not found for tile: {tile_name}")
            continue

        print(f"\n[INFO] Processing tile: {tile_name}")

        classified, posteriors, json_file = find_required_files(tile_folder)

        if not all([classified, posteriors, json_file]):
            print(f"[WARNING] Missing required files for tile {tile_name}")
            print(f"  classified: {classified}")
            print(f"  posteriors: {posteriors}")
            print(f"  json      : {json_file}")
            continue

        # Create tile folder path on S3
        s3_tile_path = f"{s3_base}/{tile_name}/"

        print(f"[INFO] Uploading to S3 folder: {s3_tile_path}")

        # Upload files
        upload_file(classified, s3_tile_path + classified.name, dry_run)
        upload_file(posteriors, s3_tile_path + posteriors.name, dry_run)
        upload_file(json_file, s3_tile_path + json_file.name, dry_run)

        print(f"[SUCCESS] Tile {tile_name} uploaded.")

    print("\nAll requested tiles processed.\n")


if __name__ == "__main__":
    main()

# To run, execute (when you want to execute it officially, do not use --dry-run):
# python -m src.upload_selected_tiles_to_s3 \
# --local-root /home/silvia/Desktop/GIGI/ESA_CCI_PROJECT/tiles_io_egeos/output/static/2019/Amazon \
# --s3-base-path s3://cci-hrlc-phase-2/data/posteriors/sar/Sentinel-1/2019 \
# --tiles 19LEL 20NQG 20NQJ 18MYS \
# --dry-run
