# src/utils/normalization.py

import numpy as np

def norm_s1_values(
    arr: np.ndarray,
    db_min: float = -25.0,
    db_max: float = 0.0,
    maxmin_start: int = 4,
    maxmin_step: int = 7,
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """
    Normalize SAR features (channel-first).

    - Standard dB channels: fixed range [db_min, db_max]
    - MAXMIN channels: percentile-based normalization

    Parameters
    ----------
    arr : np.ndarray
        Shape (C, H, W)
    """

    arr = arr.astype(np.float32)
    C, H, W = arr.shape

    out = np.zeros_like(arr, dtype=np.float32)

    maxmin_channels = set(range(maxmin_start, C, maxmin_step))

    for c in range(C):

        band = arr[c]

        # ---------------------------
        # MAXMIN features
        # ---------------------------
        if c in maxmin_channels:
            lo = np.percentile(band, p_low)
            hi = np.percentile(band, p_high)

            if hi > lo:
                band = np.clip(band, lo, hi)
                band = (band - lo) / (hi - lo)
            else:
                band = np.zeros_like(band)

        # ---------------------------
        # Standard dB features
        # ---------------------------
        else:
            band = np.clip(band, db_min, db_max)
            band = (band - db_min) / (db_max - db_min)

        out[c] = band

    return out