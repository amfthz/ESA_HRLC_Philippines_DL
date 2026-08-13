# src/tile_inference/writer.py
import os
import rasterio
import numpy as np
import json 

def _build_nodata_mask_from_reference(feature_dir: str):

    tif_files = sorted(f for f in os.listdir(feature_dir) if f.endswith(".tif"))
    if not tif_files:
        raise RuntimeError(f"No TIFF found in {feature_dir}")

    mask = None

    for f in tif_files:
        with rasterio.open(os.path.join(feature_dir, f)) as src:
            arr = src.read(1)
            valid = np.isfinite(arr)

            if mask is None:
                # initialize mask with first feature
                mask = valid.copy()
            else:
                # accumulate validity: pixel valid if ANY feature is valid
                mask |= valid

    return mask

def write_prediction_tile(
    out_path: str,
    pred: np.ndarray,              # (H, W) uint8
    reference_profile: dict,
    feature_dir: str,
):
    """
    Write predicted class map to GeoTIFF.
    """

    profile = reference_profile.copy()
    profile.update(
        count=1,
        dtype=rasterio.uint8,
        compress="lzw",
        nodata=0,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # --------------------------------------------------
    # Apply nodata mask from reference features
    # --------------------------------------------------
    nodata_mask = _build_nodata_mask_from_reference(feature_dir)
    pred = pred.copy()
    pred[~nodata_mask] = 0

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(pred.astype(np.uint8), 1)

    print(f"[WRITER] Class prediction written to: {out_path}")

def write_posteriors_uint8_tif(
    out_path: str,
    prob_u8: np.ndarray,          # (K, H, W) uint8
    reference_profile: dict,
    feature_dir: str,
):
    """
    Write posterior probability maps (uint8) to GeoTIFF.
    Values are expected in [1,255], nodata=0.
    """

    profile = reference_profile.copy()
    profile.update(
        count=prob_u8.shape[0],
        dtype=rasterio.uint8,
        nodata=0,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=512,
        blockysize=512,
        BIGTIFF="IF_NEEDED",
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # --------------------------------------------------
    # Apply nodata mask from reference features
    # --------------------------------------------------
    nodata_mask = _build_nodata_mask_from_reference(feature_dir)
    prob_u8 = prob_u8.copy()

    # Set ALL bands to 0 where nodata in features
    prob_u8[:, ~nodata_mask] = 0

    # For nodata pixels: force class-0 posterior to max uint8 (255)
    # Band 1 == class 0 in legend
    prob_u8[0, ~nodata_mask] = 255

    with rasterio.open(out_path, "w", **profile) as dst:
        for k in range(prob_u8.shape[0]):
            dst.write(prob_u8[k], k + 1)

    print(f"[WRITER] Posteriors (uint8) written to: {out_path}")

def write_posteriors_json(path, meta: dict):
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)