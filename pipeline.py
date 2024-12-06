import os
import ctypes
from tempfile import NamedTemporaryFile
from scipy.optimize import minimize_scalar

import numpy as np

import numpy as np
from scipy.optimize import minimize_scalar


def mapping3d(h, l):
    cum_sum = 0
    t = np.zeros_like(h, dtype=np.int32)
    for i in range(l):
        cum_sum += h[i]
        t[i] = np.ceil((l - 1) * cum_sum + 0.5)
    return t


def f3d(lam, h_i, h_u, l):
    h_tilde = 1 / (1 + lam) * h_i + lam / (1 + lam) * h_u
    t = mapping3d(h_tilde, l)
    d = 0
    for i in range(l):
        for j in range(i + 1):
            if h_tilde[i] > 0 and h_tilde[j] > 0 and t[i] == t[j]:
                d = max(d, i - j)
    return d


def apply_glcae_3d(volume, l=256):
    # Handle NaN and infinity values
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure volume is in valid range before histogram calculation
    volume_scaled = np.clip(volume, 0, l - 1)

    # Calculate histogram
    h_i = np.bincount(volume_scaled.astype(np.uint8).flatten())
    h_i = np.concatenate((h_i, np.zeros(l - h_i.shape[0]))) / volume_scaled.size
    h_u = np.ones_like(h_i) * 1 / l

    # Optimize lambda
    result = minimize_scalar(f3d, method="brent", args=(h_i, h_u, l))

    # Calculate mapping
    h_tilde = 1 / (1 + result.x) * h_i + result.x / (1 + result.x) * h_u
    t = mapping3d(h_tilde, l)

    # Apply mapping
    mapped = np.apply_along_axis(lambda x: t[x.astype(np.uint8)], 0, volume_scaled)

    # Scale output to 0-1 range as float32
    output = mapped.astype(np.float32)
    if output.max() != output.min():
        output = (output - output.min()) / (output.max() - output.min())
    else:
        output = np.zeros_like(output, dtype=np.float32)

    return output
