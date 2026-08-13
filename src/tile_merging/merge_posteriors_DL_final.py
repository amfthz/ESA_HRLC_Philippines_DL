#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_posteriors_DL_full_FINAL.py

This script merges:
  - DL posteriors GeoTIFF (Byte-coded, 0=NoData, valid 1..255)
  - DL classified GeoTIFF (Byte, has its own NoData)
  - an external water map (Byte/int), driven by a config JSON

It supports TWO configurations exactly as requested:

MODE A) SEASONALITY (remove_permanent missing or false)
  - Read band positions from config:
      B_WATER_SEASONAL  = water.seasonal.position
      B_WATER_PERMANENT = water.permanent.position
  - Output posteriors: input_bands + 1 (e.g., 11 -> 12)
  - ICE (input band 11) is moved to the LAST output band (e.g., band 12)
  - Water posteriors are injected in their configured positions (seasonal/permanent)
  - Non-water bands are rescaled to the remaining budget (100 - water_budget)
  - Classified output is overwritten by water classes:
      water==1 -> seasonal.cls_value
      water==2 -> permanent.cls_value    (if has_seasonal True)
      if has_seasonal False: water==1 treated as permanent only

MODE B) NO SEASONALITY (remove_permanent == true)
  - Read UNIQUE water band position from config:
      B_WATER = water.seasonal.position
  - Output posteriors: same number of bands as input (e.g., 11 -> 11)
  - The UNIQUE water posterior overwrites band B_WATER
  - ICE stays where it is in the input (band 11 stays band 11)
  - Non-water bands (all except B_WATER) are rescaled to (100 - water_budget)
  - Classified output overwritten where water is present (simple rule: water_map > 0):
      class value = water.seasonal.cls_value (as per your no-seas JSON)

IMPORTANT:
  - We do NOT enforce any exact per-pixel sum in byte space (rounding is expected).
  - Output posteriors remain Byte-coded 1..255 (0 reserved for NoData),
    compatible with downstream fusion where 0 indicates NoData.

Posterior coding:
  - Input byte 1..255 -> unit prob in [0..1] via (byte-1)/254
  - Output percent [0..100] -> byte 1..255 via round(254*perc/100 + 1)
  - NoData pixels (from classified map) -> 0 in ALL output posterior bands
"""

import argparse
import sys
import os
import datetime
import time
import json
import numpy as np
import subprocess
import shutil

from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly, GDT_Byte

BLOCK_SIZE = 512
CONFIG_FILE_PATH = "merge_config.json"

IN_BANDS_EXPECTED = 11
ICE_IN_BAND = 11  # fixed by your pipeline


# --------------------------
# Utilities
# --------------------------
def clip_images(image_paths):
    """Clip all rasters to their common intersection (optional)."""
    tmp_dir = os.path.join(os.path.dirname(image_paths[0]), ".tmp_clip")
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    shape_list = []
    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        shp_path = os.path.join(tmp_dir, f"{img_name}.shp")
        subprocess.run(["gdaltindex", shp_path, img_path], check=False)
        shape_list.append(shp_path)

    clip_shp = os.path.join(tmp_dir, "clip.shp")
    clip_tmp = os.path.join(tmp_dir, "clip_temp.shp")

    shp0 = shape_list[0]
    for shp in shape_list[1:]:
        subprocess.run(["ogr2ogr", "-overwrite", "-clipsrc", shp, clip_tmp, shp0], check=False)
        if os.path.exists(clip_shp):
            try:
                os.remove(clip_shp)
            except Exception:
                pass
        os.rename(clip_tmp, clip_shp)
        shp0 = clip_shp

    outputs = []
    for img_path in image_paths:
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        img_ext = os.path.splitext(os.path.basename(img_path))[1]
        img_dir = os.path.dirname(img_path)
        out_path = os.path.join(img_dir, f"{img_name}_clipped{img_ext}")
        subprocess.run(["gdalwarp", "-cutline", clip_shp, "-crop_to_cutline", img_path, out_path], check=False)
        outputs.append(out_path)

    return outputs


def get_last_coord(start, block, max_size):
    """Block read size helper."""
    return max_size - start if (start + block) > max_size else block


def open_raster(path):
    """Open raster read-only or fail."""
    ds = gdal.Open(path, GA_ReadOnly)
    if ds is None:
        print(f"ERROR: could not open raster: {path}")
        sys.exit(1)
    return ds


def ensure_dir(path):
    """Ensure output dir exists and ends with os.sep."""
    os.makedirs(path, exist_ok=True)
    if not path.endswith(os.sep):
        path += os.sep
    return path


def load_json_strict(path):
    """Strict JSON loader with useful error context if malformed."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            txt = f.read()
        start = max(0, e.pos - 120)
        end = min(len(txt), e.pos + 120)
        print(f"\nERROR: Invalid JSON in: {path}")
        print(f"Reason: {e.msg} (line {e.lineno}, column {e.colno}, char {e.pos})")
        print("Context around error:")
        print("------------------------------------------------------------")
        print(txt[start:end])
        print("------------------------------------------------------------\n")
        raise


def perc_to_byte_1_255(perc):
    """Percent [0..100] -> Byte-coded posterior [1..255]."""
    return np.rint((254.0 * perc / 100.0) + 1.0).astype(np.int32)


def byte_1_255_to_unit(arr):
    """Byte-coded posterior [1..255] -> unit prob [0..1]. NoData=0 must be masked elsewhere."""
    return (arr.astype(np.float32) - 1.0) / 254.0


# --------------------------
# JSON update helpers
# --------------------------
def update_json_seas(json_data, ws_cls, wp_cls, out_bands):
    """
    SEAS mode:
      output SAR_class length = out_bands (= input_bands + 1)
      strategy:
        - take first 9 entries from input SAR_class if available
        - put ws_cls at seasonal position
        - put wp_cls at permanent position
        - put ICE class value as LAST entry (moved ice)
    NOTE: Since SAR_class is a mapping list (channel order), we keep it consistent with output band order:
          [bands 1..9] + [water seasonal] + [water permanent] + [ice]
    """
    sar_class = json_data.get("SAR_class")
    if sar_class is None:
        print("WARNING: SAR_class not found in JSON. Skipping JSON update.")
        return None

    # Try to get ice class value from input mapping (ideally index 10 for 11-band input)
    ice_value = sar_class[10] if len(sar_class) >= 11 else -1
    if len(sar_class) < 11:
        print("WARNING: SAR_class length < 11. ICE value cannot be read reliably. Using -1 placeholder.")

    base9 = sar_class[:9] if len(sar_class) >= 9 else (sar_class + [-1] * (9 - len(sar_class)))
    new_sar_class = base9 + [ws_cls, wp_cls, ice_value]

    json_data["SAR_class"] = new_sar_class
    json_data["channel_num"] = out_bands
    return json_data


def update_json_no_seas(json_data, water_cls, out_bands, water_pos_1based):
    """
    NO-SEAS mode:
      output SAR_class length = out_bands (= input_bands)
      strategy:
        - keep SAR_class as-is if possible
        - ensure the water band position has value water_cls
        - do NOT move ICE (it stays in its input position)
    If SAR_class is shorter, we minimally pad with -1 to reach out_bands.
    """
    sar_class = json_data.get("SAR_class")
    if sar_class is None:
        print("WARNING: SAR_class not found in JSON. Skipping JSON update.")
        return None

    if len(sar_class) < out_bands:
        sar_class = sar_class + [-1] * (out_bands - len(sar_class))
        print(f"WARNING: SAR_class length was shorter than {out_bands}. Padding with -1 placeholders.")

    idx = water_pos_1based - 1
    if 0 <= idx < len(sar_class):
        sar_class[idx] = water_cls

    json_data["SAR_class"] = sar_class
    json_data["channel_num"] = out_bands
    return json_data


# --------------------------
# Main
# --------------------------
def main(args):
    post_path = args.input
    cls_path = args.classifiedinput
    cls_json_path = args.classifiedjson
    out_dir = ensure_dir(args.output)
    cfg_path = args.config
    block_size = int(args.blocksize)

    # --------------------------
    # Read config
    # --------------------------
    cfg = load_json_strict(cfg_path)
    water_cfg = cfg.get("water")
    if water_cfg is None:
        print("ERROR: 'water' section not found in config.")
        sys.exit(1)

    water_path = water_cfg.get("image_path")
    if not water_path:
        print("ERROR: water.image_path not found in config.")
        sys.exit(1)

    # Mode selection:
    # - NO-SEAS mode when remove_permanent == True
    # - SEAS mode otherwise (remove_permanent missing -> default False)
    remove_permanent = bool(water_cfg.get("remove_permanent", False))
    no_seas_mode = remove_permanent

    # has_seasonal is used only in SEAS mode for mask interpretation
    has_seasonal = bool(water_cfg.get("has_seasonal", False))

    seasonal_cfg = water_cfg.get("seasonal")
    permanent_cfg = water_cfg.get("permanent")
    if seasonal_cfg is None or permanent_cfg is None:
        print("ERROR: water.seasonal / water.permanent missing in config.")
        sys.exit(1)

    # Read positions from config
    pos_seasonal = int(seasonal_cfg.get("position"))
    pos_permanent = int(permanent_cfg.get("position"))

    # Percent probabilities
    ws_p = float(seasonal_cfg.get("1"))
    ws_np = float(seasonal_cfg.get("0"))

    wp_p = float(permanent_cfg.get("1"))
    wp_np = float(permanent_cfg.get("0"))

    # Class values used in classified overwrite
    ws_cls = int(seasonal_cfg.get("cls_value"))
    wp_cls = int(permanent_cfg.get("cls_value"))

    # --------------------------
    # Optional clipping
    # --------------------------
    if args.clip:
        post_path, cls_path, water_path = clip_images([post_path, cls_path, water_path])

    # --------------------------
    # Open rasters
    # --------------------------
    cls_ds = open_raster(cls_path)
    cls_band = cls_ds.GetRasterBand(1)

    post_ds = open_raster(post_path)
    in_bands = post_ds.RasterCount
    if in_bands != IN_BANDS_EXPECTED:
        print(f"ERROR: Input posteriors must have {IN_BANDS_EXPECTED} bands, got {in_bands}.")
        sys.exit(1)

    water_ds = open_raster(water_path)
    water_band = water_ds.GetRasterBand(1)

    # Use common minimum size (safer if rasters differ slightly)
    rows = min(cls_ds.RasterYSize, post_ds.RasterYSize, water_ds.RasterYSize)
    cols = min(cls_ds.RasterXSize, post_ds.RasterXSize, water_ds.RasterXSize)

    # --------------------------
    # Define output band count and dynamic band mapping
    # --------------------------
    if no_seas_mode:
        # NO-SEAS: output has SAME band count as input
        out_bands = in_bands

        # Unique water band position (read from config seasonal.position)
        B_WATER = pos_seasonal

        # ICE stays where it is: ICE_IN_BAND remains ICE_IN_BAND in output
        ICE_OUT_BAND = ICE_IN_BAND

    else:
        # SEAS: output has +1 band (ICE moved to the end)
        out_bands = in_bands + 1

        # Water band positions come from config
        B_WATER_SEASONAL = pos_seasonal
        B_WATER_PERMANENT = pos_permanent

        # ICE is moved to last band
        ICE_OUT_BAND = out_bands

    # --------------------------
    # Create outputs
    # --------------------------
    driver = post_ds.GetDriver()

    out_post_name = os.path.basename(post_path)
    out_post_ds = driver.Create(os.path.join(out_dir, out_post_name), cols, rows, out_bands, GDT_Byte)
    if out_post_ds is None:
        print(f"ERROR: Could not create output posteriors in {out_dir}")
        sys.exit(1)

    out_cls_name = os.path.basename(cls_path)
    out_cls_ds = driver.Create(os.path.join(out_dir, out_cls_name), cols, rows, 1, GDT_Byte)
    if out_cls_ds is None:
        print(f"ERROR: Could not create output classified in {out_dir}")
        sys.exit(1)
    out_cls_band = out_cls_ds.GetRasterBand(1)

    print("START AT " + str(datetime.datetime.now()) + "\n")
    t0 = time.time()

    nodata_cls = cls_band.GetNoDataValue()

    # --------------------------
    # Block-wise processing
    # --------------------------
    for y0 in range(0, rows, block_size):
        y_size = get_last_coord(y0, block_size, rows)

        for x0 in range(0, cols, block_size):
            x_size = get_last_coord(x0, block_size, cols)

            # Read blocks
            cls_block = cls_band.ReadAsArray(x0, y0, x_size, y_size).astype(np.int32)
            w_block = water_band.ReadAsArray(x0, y0, x_size, y_size).astype(np.int32)

            # NoData mask ONLY from classified (as required downstream)
            nan1 = np.isnan(cls_block)
            nan2 = (cls_block == nodata_cls) if nodata_cls is not None else np.zeros_like(nan1, dtype=bool)
            nodata_mask = np.logical_or(nan1, nan2)
            valid_mask = ~nodata_mask

            # ---------------------------------------------------------
            # Output classified map (WATER WINS)
            # ---------------------------------------------------------
            out_cls_block = cls_block.copy()

            if no_seas_mode:
                # NO-SEAS: simple water presence mask
                water_mask = (w_block > 0)
                # Use ws_cls as the water class value (matches your no-seas JSON)
                out_cls_block[water_mask] = ws_cls
            else:
                # SEAS: interpret masks using has_seasonal flag
                if has_seasonal:
                    ws_mask = (w_block == 1)
                    wp_mask = (w_block == 2)
                else:
                    ws_mask = np.zeros_like(valid_mask, dtype=bool)
                    wp_mask = (w_block == 1)

                out_cls_block[ws_mask] = ws_cls
                out_cls_block[wp_mask] = wp_cls

            # Restore NoData in classified
            if nodata_cls is not None:
                out_cls_block[nodata_mask] = nodata_cls
            out_cls_band.WriteArray(out_cls_block, x0, y0)

            # ---------------------------------------------------------
            # Output posteriors
            # ---------------------------------------------------------
            if no_seas_mode:
                # =========================
                # NO-SEAS POSTERIORS (same band count as input)
                # =========================

                # Unique water percent posterior
                water_mask = (w_block > 0)
                water_perc = np.full((y_size, x_size), ws_np, dtype=np.float32)
                water_perc[water_mask] = ws_p

                water_budget = water_perc
                if np.any((water_budget > 100.0) & valid_mask):
                    print("WARNING: water_budget > 100% found (NO-SEAS). Clamping to 100%. Check config.")
                    water_budget = np.minimum(water_budget, 100.0)

                nonwater_budget = (100.0 - water_budget).astype(np.float32)

                # Prepare output stack
                out_stack = np.zeros((out_bands, y_size, x_size), dtype=np.int32)

                def write_scaled(in_band_idx, out_band_idx):
                    """Rescale input band to non-water budget and write to out_stack."""
                    in_b = post_ds.GetRasterBand(in_band_idx).ReadAsArray(x0, y0, x_size, y_size).astype(np.int32)
                    p_unit = byte_1_255_to_unit(in_b.astype(np.float32))
                    p_unit = np.clip(p_unit, 0.0, 1.0)

                    p_perc = nonwater_budget * p_unit
                    p_perc[nodata_mask] = 0.0

                    bb = perc_to_byte_1_255(p_perc)
                    bb[nodata_mask] = 0
                    out_stack[out_band_idx - 1] = bb

                # Write all bands:
                # - overwrite ONLY the configured water band (B_WATER)
                # - all other bands come from rescaled originals
                for b in range(1, out_bands + 1):
                    if b == B_WATER:
                        # Inject water posterior into the configured band
                        w_byte = perc_to_byte_1_255(water_perc)
                        w_byte[nodata_mask] = 0
                        out_stack[b - 1] = w_byte
                    else:
                        # Rescale original band b into output band b
                        write_scaled(b, b)

                # Write to disk
                for ob in range(1, out_bands + 1):
                    out_post_ds.GetRasterBand(ob).WriteArray(out_stack[ob - 1], x0, y0)
                    out_post_ds.GetRasterBand(ob).FlushCache()

            else:
                # =========================
                # SEAS POSTERIORS (input + 1 bands)
                # =========================

                # Masks for injection
                if has_seasonal:
                    ws_mask = (w_block == 1)
                    wp_mask = (w_block == 2)
                else:
                    ws_mask = np.zeros_like(valid_mask, dtype=bool)
                    wp_mask = (w_block == 1)

                ws_perc = np.full((y_size, x_size), ws_np, dtype=np.float32)
                ws_perc[ws_mask] = ws_p

                wp_perc = np.full((y_size, x_size), wp_np, dtype=np.float32)
                wp_perc[wp_mask] = wp_p

                water_budget = ws_perc + wp_perc
                if np.any((water_budget > 100.0) & valid_mask):
                    print("WARNING: water_budget > 100% found (SEAS). Clamping to 100%. Check config.")
                    water_budget = np.minimum(water_budget, 100.0)

                nonwater_budget = (100.0 - water_budget).astype(np.float32)

                out_stack = np.zeros((out_bands, y_size, x_size), dtype=np.int32)

                def write_scaled(in_band_idx, out_band_idx):
                    """Rescale input band to non-water budget and write to out_stack."""
                    in_b = post_ds.GetRasterBand(in_band_idx).ReadAsArray(x0, y0, x_size, y_size).astype(np.int32)
                    p_unit = byte_1_255_to_unit(in_b.astype(np.float32))
                    p_unit = np.clip(p_unit, 0.0, 1.0)

                    p_perc = nonwater_budget * p_unit
                    p_perc[nodata_mask] = 0.0

                    bb = perc_to_byte_1_255(p_perc)
                    bb[nodata_mask] = 0
                    out_stack[out_band_idx - 1] = bb

                # For SEAS output, we must map input bands to output bands:
                # - Output includes two injected water bands at configured positions
                # - ICE is moved to the last band (ICE_OUT_BAND)
                #
                # We keep output band order identical to:
                #   [input 1..9] + [seasonal water] + [permanent water] + [ice moved]
                #
                # This matches your configs (positions 10 and 11) and your pipeline expectation.

                # 1..9: rescaled from input 1..9
                for b in range(1, 10):
                    write_scaled(b, b)

                # Seasonal water injected at its configured position
                ws_byte = perc_to_byte_1_255(ws_perc)
                ws_byte[nodata_mask] = 0
                out_stack[B_WATER_SEASONAL - 1] = ws_byte

                # Permanent water injected at its configured position
                wp_byte = perc_to_byte_1_255(wp_perc)
                wp_byte[nodata_mask] = 0
                out_stack[B_WATER_PERMANENT - 1] = wp_byte

                # ICE moved to last band: rescale input ICE band 11
                write_scaled(ICE_IN_BAND, ICE_OUT_BAND)

                # Write to disk
                for ob in range(1, out_bands + 1):
                    out_post_ds.GetRasterBand(ob).WriteArray(out_stack[ob - 1], x0, y0)
                    out_post_ds.GetRasterBand(ob).FlushCache()

    # --------------------------
    # Finalize output metadata
    # --------------------------
    for b in range(1, out_bands + 1):
        out_post_ds.GetRasterBand(b).SetNoDataValue(0)

    # Use classified georeferencing (as in your workflow)
    out_post_ds.SetGeoTransform(cls_ds.GetGeoTransform())
    out_post_ds.SetProjection(cls_ds.GetProjection())

    out_cls_band.FlushCache()
    if nodata_cls is not None:
        out_cls_band.SetNoDataValue(nodata_cls)
    out_cls_ds.SetGeoTransform(cls_ds.GetGeoTransform())
    out_cls_ds.SetProjection(cls_ds.GetProjection())

    t1 = time.time()
    print("\nFINISHED AT " + str(datetime.datetime.now()))
    print(f"Elapsed time: {t1 - t0:.1f} s")

    # --------------------------
    # Update JSON
    # --------------------------
    json_data = load_json_strict(cls_json_path)

    if no_seas_mode:
        # NO-SEAS: keep same number of channels and overwrite class value at water band position
        updated = update_json_no_seas(json_data, water_cls=ws_cls, out_bands=out_bands, water_pos_1based=pos_seasonal)
    else:
        # SEAS: output +1 channel, water classes inserted, ICE moved to last
        updated = update_json_seas(json_data, ws_cls=ws_cls, wp_cls=wp_cls, out_bands=out_bands)

    if updated is not None:
        out_json_name = os.path.basename(cls_json_path)
        out_json_path = os.path.join(out_dir, out_json_name)
        with open(out_json_path, "w", encoding="utf-8") as f:
            json.dump(updated, f)
        print(f"Updated JSON written to: {out_json_path}")
    else:
        print("JSON update skipped.")

    print(f"\nWrote outputs to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge DL posteriors with external water product (handles SEAS and NO-SEAS configs via remove_permanent)."
    )
    parser.add_argument("-i", "--input", required=True, help="Posteriors image path (11 bands)", metavar="INPUT_IMAGE_PATH")
    parser.add_argument("-ci", "--classifiedinput", required=True, help="Classified image path", metavar="CLASSIFIED_IMAGE_PATH")
    parser.add_argument("-j", "--classifiedjson", required=True, help="Classified JSON path", metavar="CLASSIFIED_JSON_PATH")
    parser.add_argument("-o", "--output", required=True, help="Output directory", metavar="OUTPUT_DIR_PATH")
    parser.add_argument(
        "-c", "--config", default=CONFIG_FILE_PATH, const=CONFIG_FILE_PATH, type=str, nargs="?",
        help="Config file path (default: merge_config.json)", metavar="CONFIG_FILE_PATH"
    )
    parser.add_argument("-cl", "--clip", action="store_true", help="Clip all rasters to common intersection")
    parser.add_argument(
        "-b", "--blocksize", default=BLOCK_SIZE, const=BLOCK_SIZE, type=int, nargs="?",
        help=f"Block size for raster processing (default: {BLOCK_SIZE})"
    )
    args = parser.parse_args()

    main(args)

# ============================================================
# EXAMPLE USAGE (COMMENTED)
# ============================================================
#
# The script is launched from command line.
# Paths below are EXAMPLES ONLY and must be adapted.
#
# ------------------------------------------------------------
# MODE A — SEASONALITY
# (remove_permanent = false or missing in merge_config.json)
# ------------------------------------------------------------
#
# In this mode:
# - seasonal + permanent water are both used
# - output posteriors have (input_bands + 1) bands
# - ICE class is moved to the last band
"""
python -m src.tile_merging.merge_posteriors_DL_final \
  -i  src/tile_inference/DL/static/2021/Amazon/22MCT/S1/S1_22MCT_2021_posteriors.tif \
  -ci src/tile_inference/DL/static/2021/Amazon/22MCT/S1/S1_22MCT_2021_classified.tif \
  -j  src/tile_inference/DL/static/2021/Amazon/22MCT/S1/S1_22MCT_2021_posteriors.json \
  -o  src/tile_merging/static/2021/Amazon/22MCT/S1/seasonality/ \
  -c  src/tile_inference/RF_water/static/2021/Amazon/22MCT/S1/water_seasonality_input/merge_config.json
"""
# ------------------------------------------------------------
# MODE B — NO SEASONALITY
# (remove_permanent = true in merge_config.json)
# ------------------------------------------------------------
#
# In this mode:
# - a single water contribution is used
# - output posteriors keep the SAME number of bands as input
# - ICE class stays in its original position
#
"""
python -m src.tile_merging.merge_posteriors_DL_final \
  -i  src/tile_inference/DL/static/2021/Amazon/22MCT/S1/S1_22MCT_2021_posteriors.tif \
  -ci src/tile_inference/DL/static/2021/Amazon/22MCT/S1/S1_22MCT_2021_classified.tif \
  -j  src/tile_inference/DL/static/2021/Amazon/22MCT/S1/S1_22MCT_2021_posteriors.json \
  -o  src/tile_merging/static/2021/Amazon/22MCT/S1/no_seasonality/ \
  -c  src/tile_inference/RF_water/static/2021/Amazon/22MCT/S1/water_no_seasonality_input/merge_config_no_seasonality.json
"""
#
#
# ------------------------------------------------------------
# NOTES
# ------------------------------------------------------------
# - Input posteriors must be BYTE-coded:
#       0        -> NoData
#       1 .. 255 -> probabilities
#
# - The active merging mode is controlled ONLY by the JSON
#   configuration file (merge_config*.json).
#
# - JSON metadata is automatically updated and written
#   in the output directory.
#
# ============================================================