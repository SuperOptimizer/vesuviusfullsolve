import os
import ctypes
from tempfile import NamedTemporaryFile
from scipy.optimize import minimize_scalar

import numpy as np

import numpy as np
from scipy.optimize import minimize_scalar

import numpy as np
from scipy.optimize import minimize_scalar


def mapping3d(h, l):
    """
    Compute mapping function for histogram equalization.
    Now optimized for uint8 output.
    """
    cum_sum = 0
    t = np.zeros_like(h, dtype=np.uint8)  # Changed to uint8
    for i in range(l):
        cum_sum += h[i]
        t[i] = np.clip(np.ceil((l - 1) * cum_sum + 0.5), 0, 255)  # Ensure uint8 range
    return t


def f3d(lam, h_i, h_u, l):
    """
    Objective function for lambda optimization.
    Unchanged as this operates on normalized histograms.
    """
    h_tilde = 1 / (1 + lam) * h_i + lam / (1 + lam) * h_u
    t = mapping3d(h_tilde, l)
    d = 0
    for i in range(l):
        for j in range(i + 1):
            if h_tilde[i] > 0 and h_tilde[j] > 0 and t[i] == t[j]:
                d = max(d, i - j)
    return d


def apply_glcae_3d(volume, l=256):
    """
    Apply GLCAE to a 3D volume.

    Parameters:
    -----------
    volume : ndarray
        Input 3D volume as uint8 with values in [0,255]
    l : int
        Number of intensity levels (default 256 for uint8)

    Returns:
    --------
    output : ndarray
        Contrast-enhanced volume as uint8 with values in [0,255]
    """
    # Input validation
    if volume.dtype != np.uint8:
        raise ValueError("Input volume must be uint8")

    # Handle NaN and infinity values
    volume = np.nan_to_num(volume, nan=0, posinf=255, neginf=0).astype(np.uint8)

    # Calculate histogram (no need for scaling since input is already uint8)
    h_i = np.bincount(volume.flatten(), minlength=l)
    h_i = h_i.astype(np.float32) / volume.size  # Normalize
    h_u = np.ones_like(h_i, dtype=np.float32) * (1.0 / l)

    # Optimize lambda
    result = minimize_scalar(f3d, method="brent", args=(h_i, h_u, l))

    # Calculate mapping
    h_tilde = 1 / (1 + result.x) * h_i + result.x / (1 + result.x) * h_u
    t = mapping3d(h_tilde, l)

    # Apply mapping (optimized for uint8)
    output = t[volume]  # Direct indexing works since input is uint8

    return output