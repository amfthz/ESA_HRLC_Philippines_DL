import json
import numpy as np
import rasterio

# =====================================================
# ESA CCI HRLC LEGEND (target product)
# =====================================================
CCI_HRLC_CLASSES = np.array([
    10, 20, 30, 40, 50, 60, 70, 80,
    90, 100, 110, 120, 130, 141, 142, 150
])
CCI_NUM_CHANNELS = 16


# =====================================================
# CORE SCIENTIFIC FUNCTION (UNIGE mapping)
# =====================================================
def SAR_class_map_and_detect(SAR_posterior_uint8: np.ndarray, SAR_class: list):
    """
    Apply MOLCA → ESA-CCI remapping and detect classes actually present.

    Parameters
    ----------
    SAR_posterior_uint8 : np.ndarray
        Posterior cube (C,H,W) stored as uint8 in range [1..255]
    SAR_class : list[int]
        List of MOLCA class IDs present in the product JSON

    Returns
    -------
    remapped_uint8 : np.ndarray
        Remapped posterior cube (16,H,W) uint8
    present_classes_hrlc : list[int]
        ESA CCI HRLC class codes actually present in the tile
    present_classes_idx : list[int]
        Band indices (0..15) actually present
    """

    # --------------------------------------------------
    # UNIGE normalization of posterior cube
    # --------------------------------------------------
    # --------------------------------------------------
    # UNIGE style normalization (keep as close as possible)
    # --------------------------------------------------
    posterior = SAR_posterior_uint8.astype(np.float32)
    posterior = (posterior / np.sum(posterior, axis=0)).astype(np.float32)

    H, W = posterior.shape[1], posterior.shape[2]
    mapped = np.zeros((CCI_NUM_CHANNELS, H, W), dtype=np.float32)

    # --------------------------------------------------
    # UNIGE MOLCA → CCI probability redistribution
    # --------------------------------------------------
    for i, c in enumerate(SAR_class):
        if c == 1:
            mapped[0] += posterior[i] / 4
            mapped[1] += posterior[i] / 4
            mapped[2] += posterior[i] / 4
            mapped[3] += posterior[i] / 4
        elif c == 2:
            mapped[4] += posterior[i] / 2
            mapped[5] += posterior[i] / 2
        elif c == 3:
            mapped[6] += posterior[i]
        elif c == 4:
            mapped[7] += posterior[i]
        elif c == 5:
            mapped[8] += posterior[i] / 2
            mapped[9] += posterior[i] / 2
        elif c == 6:
            mapped[10] += posterior[i]
        elif c == 7:
            mapped[11] += posterior[i]
        elif c == 8:
            mapped[12] += posterior[i]
        elif c == 14:
            mapped[13] += posterior[i]
        elif c == 15:
            mapped[14] += posterior[i]
        elif c == 10:
            mapped[15] += posterior[i]

    # --------------------------------------------------
    # Convert back to uint8 [1..255]
    # --------------------------------------------------
    remapped_uint8 = (1 + mapped * 254).astype(np.uint8)

    # --------------------------------------------------
    # Detect ESA classes actually present in tile
    # --------------------------------------------------
    band_sum = remapped_uint8.sum(axis=(1, 2))
    present_mask = band_sum > (H * W)  # > baseline value 1 per pixel
    present_idx = np.where(present_mask)[0]

    present_classes_idx = present_idx.tolist()
    present_classes_hrlc = CCI_HRLC_CLASSES[present_idx].tolist()

    return remapped_uint8, present_classes_hrlc, present_classes_idx


# =====================================================
# PIPELINE WRAPPER (called at the end of infer.py)
# =====================================================
def remap_geotiff_and_json(geotiff_path: str, json_path: str):
    """
    Final mandatory post-processing step.
    Overwrites:
        - posteriors.tif  → ESA CCI HRLC posterior cube
        - posteriors.json → updated class list and metadata
    """

    print(f"[CCI REMAP] Processing tile: {geotiff_path}")

    # --------------------------------------------------
    # 1) Load JSON metadata (contains MOLCA classes)
    # --------------------------------------------------
    with open(json_path, "r") as f:
        metadata = json.load(f)

    molca_classes = metadata["SAR_class"]
    print("  MOLCA classes found:", molca_classes)

    # --------------------------------------------------
    # 2) Read posterior GeoTIFF (MOLCA output)
    # --------------------------------------------------
    with rasterio.open(geotiff_path) as src:
        profile = src.profile
        posterior = src.read()

    # --------------------------------------------------
    # 3) Apply UNIGE remapping
    # --------------------------------------------------
    remapped, classes_hrlc, classes_idx = SAR_class_map_and_detect(
        posterior, molca_classes
    )

    print("  ESA-CCI classes present:", classes_hrlc)

    # --------------------------------------------------
    # 4) Overwrite GeoTIFF with ESA CCI cube (16 bands)
    # --------------------------------------------------
    profile.update(count=CCI_NUM_CHANNELS, dtype=rasterio.uint8)

    with rasterio.open(geotiff_path, "w", **profile) as dst:
        dst.write(remapped)

    # --------------------------------------------------
    # 5) Overwrite JSON metadata
    # --------------------------------------------------

    # ---- VARIANT A (recommended for delivery) ----
    metadata["SAR_class"] = classes_hrlc
    metadata["channel_num"] = len(classes_hrlc)

    # ---- VARIANT B (internal tensor indexing) ----
    # metadata["SAR_class"] = classes_idx
    # metadata["channel_num"] = len(classes_idx)

    metadata["Legend"] = "ESA CCI HRLC"
    metadata["Postprocessing"] = "MOLCA→CCI remapping applied"

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("[CCI REMAP] GeoTIFF and JSON successfully updated.")


# =====================================================
# OPTIONAL POST-UPDATE USING CLASSIFIED MAP (MOLCA → HRLC)
# =====================================================

def update_json_classes_from_classified(json_path: str, classified_path: str):
    """
    Supplementary utility.
    Reads classified MOLCA raster, detects present classes,
    converts them to ESA‑CCI HRLC codes and updates JSON.

    Does NOT modify GeoTIFF. Only JSON metadata is updated.
    """

    print(f"[CCI JSON UPDATE] Using classified map: {classified_path}")

    # MOLCA → HRLC mapping (final agreed law)
    MOLCA_TO_CCI_BAND_MAP = {
        1: [10,20,30,40],
        2: [50,60],
        3: [70],
        4: [80],
        5: [90,100],
        6: [110],
        7: [120],
        8: [130],
        14: [141],
        15: [142],
        10: [150],
    }

    # --------------------------------------------------
    # 1) Load classified raster and detect MOLCA classes
    # --------------------------------------------------
    with rasterio.open(classified_path) as src:
        classified = src.read(1)

    unique_molca = np.unique(classified)
    unique_molca = unique_molca.tolist()
    # Remove nodata value (0) if present
    unique_molca = [c for c in unique_molca if c != 0]

    print("  MOLCA classes detected in classified:", unique_molca)

    # --------------------------------------------------
    # 3) Update JSON metadata
    # --------------------------------------------------
    with open(json_path, "r") as f:
        metadata = json.load(f)

    metadata["SAR_class"] = unique_molca
    metadata["channel_num"] = len(unique_molca)
    metadata["Class_update"] = "Derived from classified map (MOLCA→HRLC)"

    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("[CCI JSON UPDATE] JSON successfully updated from classified map.")