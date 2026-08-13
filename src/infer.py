#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pipeline.py

Unified pipeline for:
1) SAR tile inference with Deep Learning
2) Merging DL posteriors with external water products

The pipeline automatically detects whether the water map contains
seasonal water (0,1,2) or non‑seasonal water (0,1) and generates
an appropriate merging configuration JSON.

------------------------------------------------------------------
Example CLI usage
------------------------------------------------------------------

python -m src.infer \
  --input-dir /data/tiles_io_egeos/input/static/2019/Amazon/18MYS/S1 \
  --checkpoint /models/SAR_DL_model.pth \
  --output-base-dir /data/tiles_io_egeos/output \
  --type static \
  --year 2019 \
  --area Amazon \
  --tile 18MYS \
  --source S1 \
  --patch-size 256 \
  --stride 128 \
  --batch-size 4

------------------------------------------------------------------
Expected input directory structure
------------------------------------------------------------------

/data/tiles_io_egeos/input/static/2019/Amazon/18MYS/S1
│
├── features
│     ├── feature_01.tif
│     ├── ...
│     └── feature_28.tif
│
└── water_map
      └── 18MYS_water_*.tif

Where:
- features/ contains the 28 SAR input features used for DL inference
- water_map/ contains the external water mask used for posterior merging

The pipeline will:
  1. run DL inference using the SAR features
  2. detect water map type (seasonal vs non‑seasonal)
  3. automatically generate the correct merge_config JSON
  4. run posterior merging
  5. apply UNIGE posterior remapping
  6. optionally override sliding window parameters (patch size, stride, batch size) directly from the CLI
"""

import argparse
import copy
import os
import sys
import subprocess
import shutil

from src.tile_inference.config import load_inference_config
from src.tile_inference.SAR_class_map import remap_geotiff_and_json, update_json_classes_from_classified

import json
import numpy as np
from pathlib import Path

from osgeo import gdal

from src.utils.fill_missing_features import prepare_input_directory


def ensure_input_structure(input_dir: Path, tile_id: str):
    """
    Ensure that the tile input directory has the expected structure:

    <tile_root>/S1/
        features/
        water_map/

    The function is robust to these cases:
    - input_dir already points to .../<TILE>/S1
    - input_dir points to .../<TILE>
    - input_dir points to .../<TILE>/features
    - input_dir points to .../<TILE>/water_map

    If SAR features or water maps are found in the wrong place, they are
    automatically moved into:
        S1/features/
        S1/water_map/

    Rules:
    - SAR features start with tile_id (e.g. 18MYS_*)
    - Water map contains the word 'water' (case insensitive)

    Returns
    -------
    Path
        Resolved S1 directory.
    """

    input_dir = Path(input_dir)

    # --------------------------------------------------
    # Resolve the canonical S1 directory.
    # --------------------------------------------------
    if input_dir.name in ["features", "water_map"]:
        print(f"[PIPELINE] input_dir points to '{input_dir.name}/'. Moving to parent S1 directory.")
        s1_dir = input_dir.parent
    elif input_dir.name == "S1":
        s1_dir = input_dir
    else:
        # Standard case: input_dir points to tile directory, so create/use S1 under it.
        s1_dir = input_dir / "S1"

    s1_dir.mkdir(parents=True, exist_ok=True)

    features_dir = s1_dir / "features"
    water_dir = s1_dir / "water_map"
    features_dir.mkdir(exist_ok=True)
    water_dir.mkdir(exist_ok=True)

    # --------------------------------------------------
    # Candidate directories where misplaced TIFFs may exist.
    # We inspect both the resolved S1 directory and its parent tile directory.
    # --------------------------------------------------
    candidate_dirs = [s1_dir]
    if s1_dir.parent != s1_dir:
        candidate_dirs.append(s1_dir.parent)

    seen_files = set()

    for base_dir in candidate_dirs:
        for f in base_dir.iterdir():
            if f in seen_files:
                continue
            seen_files.add(f)

            # Skip expected folders
            if f.name in ["features", "water_map", "S1"]:
                continue

            if not f.is_file():
                continue

            if f.suffix.lower() != ".tif":
                continue

            name_lower = f.name.lower()

            # Detect water map
            if "water" in name_lower:
                target = water_dir / f.name

            # Detect SAR feature
            elif f.name.startswith(tile_id):
                target = features_dir / f.name

            else:
                continue

            if f.resolve() == target.resolve():
                continue

            if not target.exists():
                print(f"[PIPELINE] Moving {f.name} -> {target.parent}")
                f.rename(target)
            else:
                print(f"[PIPELINE] Skipping move for {f.name}: target already exists")

    return s1_dir


def detect_water_mode(water_path: Path):
    """Detect if water map contains seasonal classes."""
    ds = gdal.Open(str(water_path))
    arr = ds.GetRasterBand(1).ReadAsArray()
    unique_vals = np.unique(arr)

    if 2 in unique_vals:
        return "seasonal"
    else:
        return "no_seasonal"


def create_merge_config(water_path: Path, mode: str):
    water_path = Path(water_path).resolve()

    if not water_path.exists():
        raise RuntimeError(f"Water raster not found: {water_path}")

    if mode == "seasonal":
        config = {
            "water": {
                "image_path": str(water_path),
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
        config = {
            "water": {
                "image_path": str(water_path),
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

    out_dir = water_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    return out_path


# --------------------------------------------------
# CLI
# --------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Run DL tile inference and merge posteriors with water products"
    )

    # --- Optional overrides (fallback to YAML) ---
    p.add_argument("--input-dir", type=str, help="Input directory containing 'features/' and 'water_map/'")
    p.add_argument(
        "--polarization",
        type=str,
        choices=["VH", "VV", "VH_VV"],
        help="Polarization mode to use (overrides data.polarization_mode in YAML)"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint to use (overrides model.checkpoints[polarization])"
    )
    p.add_argument("--output-base-dir", type=str, help="Base output directory")

    p.add_argument("--type", type=str, choices=["static", "historical"])
    p.add_argument("--year", type=int)
    p.add_argument("--area", type=str)
    p.add_argument("--tile", type=str)
    p.add_argument("--source", type=str)

    # --- Sliding window inference parameters (override YAML) ---
    p.add_argument("--patch-size", type=int, help="Sliding window patch size")
    p.add_argument("--stride", type=int, help="Sliding window stride")
    p.add_argument("--batch-size", type=int, help="Sliding window batch size")

    # --- Config ---
    p.add_argument(
        "-c", "--config",
        default="configs/inference.yaml",
        help="Inference YAML config (default: inference.yaml)"
    )

    return p.parse_args()


# --------------------------------------------------
# Config override helper
# --------------------------------------------------
def override_cfg(cfg: dict, args) -> dict:
    cfg = copy.deepcopy(cfg)

    # Tile / input (CLI-driven)
    if args.input_dir:
        cfg.setdefault("cli", {})
        cfg["cli"]["input_dir"] = args.input_dir

    # Polarization override
    if args.polarization:
        cfg["data"]["polarization_mode"] = args.polarization

    # Model checkpoint override (CLI is mandatory)
    if args.checkpoint:
        if "model" not in cfg:
            raise KeyError("Missing 'model' section in config")
        cfg["model"]["checkpoint"] = args.checkpoint

    # Sliding window overrides
    if args.patch_size or args.stride or args.batch_size:
        cfg.setdefault("cli", {})

        if args.patch_size:
            cfg["cli"]["patch_size"] = args.patch_size

        if args.stride:
            cfg["cli"]["stride"] = args.stride

        if args.batch_size:
            cfg["cli"]["batch_size"] = args.batch_size

    return cfg


# --------------------------------------------------
# Paths
# --------------------------------------------------
# Deleted build_base_output_dir function as per instructions


def build_intermediate_dir(base_out: str) -> str:
    return os.path.join(base_out, "_intermediate")


# --------------------------------------------------
# DL inference (wrapper)
# --------------------------------------------------
def run_dl_inference(cfg: dict, intermediate_dir: str):
    os.makedirs(intermediate_dir, exist_ok=True)

    # Add intermediate_dir to cfg under cli
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("cli", {})
    cfg["cli"]["intermediate_dir"] = intermediate_dir

    print("[PIPELINE] Running DL inference...")
    from src.tile_inference.main import main as dl_main

    outputs = dl_main(cfg_override=cfg)

    return outputs

# --------------------------------------------------
# Merging
# --------------------------------------------------
def run_merging(
    mode: str,
    dl_outputs: dict,
    cfg: dict,
    base_out: str,
):
    """
    mode: 'seasonality' | 'no_seasonality'
    """
    merge_cfg = cfg.get("cli", {}).get(f"{mode}_config")
    if merge_cfg is None:
        raise ValueError(f"Missing CLI merge config for {mode}.")

    out_dir = os.path.join(base_out, mode)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[PIPELINE] Running merging ({mode})...")

    cmd = [
        sys.executable,
        "-m", "src.tile_merging.merge_posteriors_DL_final",
        "-i", dl_outputs["posteriors"],
        "-ci", dl_outputs["classified"],
        "-j", dl_outputs["json"],
        "-o", out_dir,
        "-c", merge_cfg,
    ]

    subprocess.run(cmd, check=True)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    args = parse_args()

    # --- Load config ---
    cfg = load_inference_config(args.config)

    # Removed BACKWARD-COMPATIBILITY ALIASES block as per instructions

    # --- Override with CLI ---
    cfg = override_cfg(cfg, args)

    # --------------------------------------------------
    # Inject CLI identifiers for downstream naming (tile_inference.main)
    # --------------------------------------------------
    cfg.setdefault("cli", {})
    cfg["cli"].update({
        "source": args.source,
        "tile": args.tile,
        "year": args.year,
        "area": args.area,
        "type": args.type,
    })

    # --- Build output dirs strictly from CLI ---
    if not all([args.output_base_dir, args.type, args.year, args.area, args.tile, args.source]):
        raise ValueError("All output identifiers must be provided via CLI.")

    base_out = os.path.join(
        args.output_base_dir,
        args.type,
        str(args.year),
        args.area,
        args.tile,
        args.source,
    )
    intermediate_dir = os.path.join(base_out, "_intermediate")

    print("\n[DEBUG] FINAL CONFIG USED FOR DL INFERENCE")
    print(f"  polarization_mode : {cfg['data']['polarization_mode']}")
    input_dir_debug = cfg.get("cli", {}).get("input_dir")
    print(f"  input_dir         : {input_dir_debug}")
    print(f"  checkpoint        : {cfg['model']['checkpoint']}")
    print(f"  output_base_dir   : {args.output_base_dir}")
    print(f"  resolved intermediate_dir : {intermediate_dir}")

    cli_cfg = cfg.get("cli", {})
    if "patch_size" in cli_cfg:
        print(f"  patch_size        : {cli_cfg['patch_size']}")
    if "stride" in cli_cfg:
        print(f"  stride            : {cli_cfg['stride']}")
    if "batch_size" in cli_cfg:
        print(f"  batch_size        : {cli_cfg['batch_size']}")

    # --------------------------------------------------
    # Prepare input directory (fill missing features and
    # create water map if missing)
    # --------------------------------------------------

    if args.input_dir is None:
        raise RuntimeError("--input-dir must be provided")

    input_dir = Path(args.input_dir)

    print("[PIPELINE] Checking input directory structure...")
    s1_dir = ensure_input_structure(input_dir, args.tile)

    # From this point on, always use the resolved canonical S1 directory
    args.input_dir = str(s1_dir)
    cfg.setdefault("cli", {})
    cfg["cli"]["input_dir"] = str(s1_dir)

    print(f"[PIPELINE] Resolved canonical input_dir: {s1_dir}")

    print("[PIPELINE] Preparing input directory...")
    prepare_input_directory(str(s1_dir))

    # --- DL inference ---
    dl_outputs = run_dl_inference(cfg, intermediate_dir)

    # --------------------------------------------------
    # Automatic water merging detection
    # --------------------------------------------------

    input_dir = Path(s1_dir)
    water_dir = input_dir / "water_map"

    # Detect water map
    water_files = list(water_dir.glob("*.tif"))
    if len(water_files) == 0:
        raise RuntimeError(f"No water map found in {water_dir}")

    water_path = water_files[0]

    mode = detect_water_mode(water_path)

    # Create JSON config automatically
    merge_config = create_merge_config(water_path, mode)

    print(f"[PIPELINE] Detected water mode: {mode}")

    # --------------------------------------------------
    # Choose output directory name based on water mode
    # --------------------------------------------------

    if mode == "seasonal":
        out_dir = os.path.join(base_out, "seasonality")
    else:
        out_dir = os.path.join(base_out, "no_seasonality")

    cmd = [
        sys.executable,
        "-m", "src.tile_merging.merge_posteriors_DL_final",
        "-i", dl_outputs["posteriors"],
        "-ci", dl_outputs["classified"],
        "-j", dl_outputs["json"],
        "-o", out_dir,
        "-c", str(merge_config),
    ]

    subprocess.run(cmd, check=True)

    # --------------------------------------------------
    # Robust search of all remapping targets under base_out
    # --------------------------------------------------
    remapped_counter = 0
    search_root = base_out

    print(f"[PIPELINE] Searching recursively for posteriors under: {search_root}")

    if not os.path.isdir(search_root):
        raise RuntimeError(f"[PIPELINE ERROR] Output directory not found: {search_root}")

    for root, dirs, files in os.walk(search_root):
        # Skip DL intermediate outputs
        if "_intermediate" in root:
            continue

        # Look for any *_posteriors.tif produced by merging
        tif_candidates = [f for f in files if f.endswith("_posteriors.tif")]

        for tif_name in tif_candidates:
            geotiff_path = os.path.join(root, tif_name)

            # Expected paired JSON has the same prefix
            json_name = tif_name.replace("_posteriors.tif", "_posteriors.json")
            json_path = os.path.join(root, json_name)

            # --------------------------------------------------
            # Rename outputs by appending _DL suffix
            # --------------------------------------------------
            new_geotiff_path = geotiff_path.replace(".tif", "_DL.tif")
            new_json_path = json_path.replace(".json", "_DL.json")

            if not os.path.exists(new_geotiff_path):
                os.rename(geotiff_path, new_geotiff_path)
                print(f"[PIPELINE] Renamed TIFF -> {new_geotiff_path}")
            else:
                print(f"[PIPELINE] _DL TIFF already exists, using existing: {new_geotiff_path}")

            if os.path.exists(json_path) and not os.path.exists(new_json_path):
                os.rename(json_path, new_json_path)
                print(f"[PIPELINE] Renamed JSON -> {new_json_path}")
            elif os.path.exists(new_json_path):
                print(f"[PIPELINE] _DL JSON already exists, using existing: {new_json_path}")

            # --------------------------------------------------
            # Rename classified map by appending _DL suffix
            # --------------------------------------------------
            classified_name_orig = tif_name.replace("_posteriors.tif", "_classified.tif")
            classified_path_orig = os.path.join(root, classified_name_orig)

            if os.path.exists(classified_path_orig):
                new_classified_path = classified_path_orig.replace(".tif", "_DL.tif")
                if not os.path.exists(new_classified_path):
                    os.rename(classified_path_orig, new_classified_path)
                    print(f"[PIPELINE] Renamed classified TIFF -> {new_classified_path}")
                else:
                    print(f"[PIPELINE] _DL classified TIFF already exists, using existing: {new_classified_path}")
            else:
                new_classified_path = None

            # Update paths to use renamed files
            geotiff_path = new_geotiff_path
            json_path = new_json_path

            print("[PIPELINE] Remapping target found:")
            print(f"  TIFF: {geotiff_path}")
            print(f"  JSON: {json_path}")

            # 1) Apply UNIGE posterior remapping
            remap_geotiff_and_json(geotiff_path, json_path)

            # 2) OPTIONAL: update JSON classes using classified map if present
            # Use already renamed classified map (_DL) if available
            if new_classified_path is not None and os.path.exists(new_classified_path):
                print(f"[PIPELINE] Updating JSON classes using classified map: {new_classified_path}")
                update_json_classes_from_classified(json_path, new_classified_path)
            else:
                print(f"[WARNING] Classified _DL map not found for {geotiff_path}, JSON classes not updated.")

            remapped_counter += 1

    if remapped_counter == 0:
        raise RuntimeError(
            "[PIPELINE ERROR] No posteriors.tif were found anywhere under base_out. "
            "Check merging output structure or filenames."
        )

    print(f"[PIPELINE] Remapping completed. Files processed: {remapped_counter}")

    # --------------------------------------------------
    # CLEANUP INTERMEDIATE FILES
    # --------------------------------------------------
    if os.path.isdir(intermediate_dir):
        print(f"[PIPELINE] Cleaning intermediate directory: {intermediate_dir}")
        shutil.rmtree(intermediate_dir)

    print("\n[PIPELINE] DONE")
    print(f"Base output directory: {base_out}")


if __name__ == "__main__":
    main()