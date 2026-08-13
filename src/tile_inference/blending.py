# src/tile_inference/blending.py
import numpy as np

def hann2d(patch_size: int, eps: float = 1e-6) -> np.ndarray:
    """
    2D Hann window, normalized in [0,1] with non-zero floor (eps).
    """
    w1 = np.hanning(patch_size).astype(np.float32)
    w2 = np.hanning(patch_size).astype(np.float32)
    w = np.outer(w1, w2)
    w = np.maximum(w, eps)
    w = w / w.max()
    return w.astype(np.float32)  # (ps,ps)