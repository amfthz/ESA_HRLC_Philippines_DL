# src/data/preprocessing.py
import numpy as np
import cv2
from typing import Dict, Tuple

def nan_to_zero(arr: np.ndarray) -> np.ndarray:
    if np.issubdtype(arr.dtype, np.floating):
        arr = arr.copy()
        arr[np.isnan(arr)] = 0
    return arr

def resize_nearest(arr, size):
    if size is None:
        return arr

    # arr: (C,H,W) or (H,W)
    if arr.ndim == 2:
        return cv2.resize(arr, size, interpolation=cv2.INTER_NEAREST)

    elif arr.ndim == 3:
        C = arr.shape[0]
        resized = [
            cv2.resize(arr[c], size, interpolation=cv2.INTER_NEAREST)
            for c in range(C)
        ]
        return np.stack(resized, axis=0)

    else:
        raise ValueError(f"Unsupported array shape for resize: {arr.shape}")

def remap_mask_values(mask: np.ndarray, raw_to_class: Dict[int, int]) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.int64)
    for raw_val, class_id in raw_to_class.items():
        out[mask == raw_val] = class_id
    return out