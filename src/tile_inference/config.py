# src/tile_inference/config.py
import yaml
from pathlib import Path


# --------------------------------------------------
# Config loader (harmonized with new inference.yaml)
# --------------------------------------------------
def load_inference_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # --------------------------------------------------
    # DATA / POLARIZATION
    # --------------------------------------------------
    data = cfg["data"]

    mode = data["polarization_mode"]  # VH | VV | VH_VV
    pol_map = data["pol_map"]
    band_mapping = data["band_mapping"]

    if mode not in band_mapping:
        raise KeyError(f"Invalid polarization_mode '{mode}'")

    # Polarizations used in this mode (e.g. ["VH","VV"])
    pols = band_mapping[mode]

    # Convert to band indices (0-based)
    band_ids = [pol_map[p] for p in pols]

    n_features = int(data["n_features"])
    n_channels = n_features * len(band_ids)

    # --------------------------------------------------
    # MODEL
    # --------------------------------------------------
    model = cfg["model"]

    expected = int(model["expected_channels"][mode])
    if expected != n_channels:
        raise ValueError(
            f"Channel mismatch for mode '{mode}': "
            f"expected {expected}, got {n_channels}"
        )

    # --------------------------------------------------
    # PATCHING SANITY CHECKS
    # --------------------------------------------------
    patching = cfg["patching"]
    ps = int(patching["patch_size"])
    st = int(patching["stride"])

    if ps % 32 != 0:
        raise ValueError("patch_size must be divisible by 32 (SwinUNETR constraint).")
    if st <= 0 or st > ps:
        raise ValueError("stride must be in (0, patch_size].")

    # --------------------------------------------------
    # Inject derived + backward-compatible fields
    # --------------------------------------------------
    cfg["polarization_mode"] = mode
    cfg["pol_map"] = pol_map
    cfg["band_mapping"] = band_mapping
    cfg["n_features"] = n_features
    cfg["bands"] = band_ids
    cfg["n_channels"] = n_channels

    cfg["normalization"] = cfg["preprocessing"]["normalization"]

    return cfg