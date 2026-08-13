# src/utils/raster_io.py

import rasterio
import numpy as np
from typing import Tuple, Dict


def read_raster(path: str) -> Tuple[np.ndarray, Dict]:
    """
    Read a raster file using rasterio.

    Parameters
    ----------
    path : str
        Path to the raster file.

    Returns
    -------
    data : np.ndarray
        Raster data.
        Shape:
          - (C, H, W) if multi-band
          - (H, W) if single-band
    meta : dict
        Raster metadata.
    """
    with rasterio.open(path) as src:
        data = src.read()      
        meta = src.meta.copy()

    if data.shape[0] == 1:
        data = data[0]

    return data, meta