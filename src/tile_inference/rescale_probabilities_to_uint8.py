#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src.tile_inference.rescale_probabilities_to_uint8.py
"""
Reads a multiband GeoTIFF containing probabilities in [0,1], rescales them to [1,255],
rounds to the nearest integer, and saves the result as uint8 while preserving
georeferencing information.
Value 0 is reserved for NoData (if nodata is not defined in the input, it is set to 0 in the output).

Usage:
  python posteriors_rescale_uint8.py -i input_prob.tif -o output_uint8.tif
"""
import numpy as np

def rescale_probabilities_to_uint8(prob: np.ndarray) -> np.ndarray:
    """
    Rescale probabilities from [0,1] to uint8 [1,255].

    Parameters
    ----------
    prob : np.ndarray
        Array (K, H, W) with float probabilities in [0,1]

    Returns
    -------
    np.ndarray
        Array (K, H, W) uint8, with values in [1,255]
        (0 reserved for NoData)
    """
    prob = np.clip(prob, 0.0, 1.0)
    out = np.round(prob * 254.0 + 1.0).astype(np.uint8)
    return out