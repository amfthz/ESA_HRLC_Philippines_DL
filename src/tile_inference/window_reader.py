# src/tile_inference/window_reader.py
import os
import rasterio
import numpy as np
from rasterio.windows import Window

class FeatureWindowReader:
    """
    Keeps feature rasters open and reads (bands, patch) windows on demand.
    """
    def __init__(self, feature_dir: str, n_features: int):
        self.feature_dir = feature_dir
        self.n_features = n_features
        self.paths = None
        self.srcs = None
        self.profile = None
        self.H = None
        self.W = None

    def __enter__(self):
        self.paths = sorted([
            os.path.join(self.feature_dir, f)
            for f in os.listdir(self.feature_dir)
            if f.lower().endswith(".tif")
        ])
        if len(self.paths) != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, found {len(self.paths)}")

        self.srcs = [rasterio.open(p) for p in self.paths]

        # reference metadata
        self.profile = self.srcs[0].profile
        self.H = self.srcs[0].height
        self.W = self.srcs[0].width

        # quick consistency check
        for s in self.srcs[1:]:
            if s.height != self.H or s.width != self.W:
                raise ValueError("Not all feature rasters have the same shape.")

        return self

    def __exit__(self, exc_type, exc, tb):
        if self.srcs:
            for s in self.srcs:
                s.close()

    def read_patch(self, r0: int, c0: int, patch_size: int, band_ids: list[int]) -> np.ndarray:
        """
        Returns patch stacked as (C, ps, ps) with C = n_features * len(band_ids),
        in feature-time order (sorted filenames) and polarization order band_ids.
        """
        ps = patch_size
        win = Window(col_off=c0, row_off=r0, width=ps, height=ps)

        # list of (P, ps, ps), one per feature
        per_feature = []
        for src in self.srcs:
            bands = []
            for b in band_ids:
                # rasterio is 1-based
                arr = src.read(b + 1, window=win).astype(np.float32)
                bands.append(arr)
            per_feature.append(np.stack(bands, axis=0))  # (P,ps,ps)

        stack = np.stack(per_feature, axis=0)   # (T,P,ps,ps)
        T, P, _, _ = stack.shape
        stack = stack.reshape(T * P, ps, ps)    # (C,ps,ps)

        # --------------------------------------------------
        # NaN handling strategy
        # --------------------------------------------------
        # If some features are NaN but others are valid for a pixel,
        # replace those NaNs with 0 so the model can still use the
        # remaining valid information.
        # If ALL features are NaN for a pixel, keep NaN so the pixel
        # can later be mapped to NoData in the output.

        all_nan_mask = np.isnan(stack).all(axis=0)  # (ps, ps)

        # replace partial NaNs with 0
        stack = np.nan_to_num(stack, nan=0.0)

        # restore NaNs where the entire stack was NaN
        if np.any(all_nan_mask):
            stack[:, all_nan_mask] = np.nan

        return stack