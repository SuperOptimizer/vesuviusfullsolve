import os
import ctypes
from tempfile import NamedTemporaryFile
from scipy.optimize import minimize_scalar

import numpy as np
from typing import Tuple
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


import numpy as np
from numba import jit
import numpy.typing as npt


@jit(nopython=True)
def analyze_1d_sequence(sequence: np.ndarray, noise_threshold: int = 30) -> np.ndarray:
    """
    Analyze a 1D sequence to identify void regions based on transitions.
    Returns a mask where 1 indicates void space and 0 indicates matter.
    """
    length = len(sequence)
    mask = np.zeros(length, dtype=np.uint8)

    # Find transitions
    i = 0
    while i < length:
        # Skip high-value regions
        while i < length and sequence[i] > noise_threshold:
            i += 1

        if i >= length:
            break

        # We've hit a potential void region
        start_idx = i
        max_val = sequence[i]

        # Analyze until we hit a high value or end
        while i < length and sequence[i] <= noise_threshold:
            max_val = max(max_val, sequence[i])
            i += 1

        # If we didn't reach a high value in this segment, mark as void
        if max_val <= noise_threshold:
            mask[start_idx:i] = 1

    return mask


@jit(nopython=True)
def process_volume(volume: np.ndarray, noise_threshold: int = 30, min_votes: int = 4) -> np.ndarray:
    """
    Process a 3D volume to identify void spaces using directional analysis.

    Args:
        volume: 3D numpy array of uint8 values
        noise_threshold: Maximum value to consider as potential void
        min_votes: Minimum number of directions (1-6) that must agree for a voxel to be considered void

    Returns:
        Binary mask where 1 indicates void space
    """
    depth, height, width = volume.shape
    votes = np.zeros_like(volume, dtype=np.uint8)

    # Process along depth (front-back)
    for y in range(height):
        for x in range(width):
            sequence = volume[:, y, x]
            mask = analyze_1d_sequence(sequence, noise_threshold)
            votes[:, y, x] += mask

    # Process along height (up-down)
    for z in range(depth):
        for x in range(width):
            sequence = volume[z, :, x]
            mask = analyze_1d_sequence(sequence, noise_threshold)
            votes[z, :, x] += mask

    # Process along width (left-right)
    for z in range(depth):
        for y in range(height):
            sequence = volume[z, y, :]
            mask = analyze_1d_sequence(sequence, noise_threshold)
            votes[z, y, :] += mask

    # Create final mask based on vote threshold
    return (votes >= min_votes).astype(np.uint8)


def segment_volume(volume: npt.NDArray[np.uint8], noise_threshold: int = 64, min_votes: int = 3) -> npt.NDArray[
    np.uint8]:
    """
    Main function to segment a 3D volume, identifying and removing void spaces.

    Args:
        volume: Input 3D volume (uint8 values 0-255)
        noise_threshold: Maximum value to consider as potential void
        min_votes: Minimum number of directions (1-6) that must agree for a voxel to be considered void

    Returns:
        Segmented volume with void spaces set to 0
    """
    # Validate input
    if volume.dtype != np.uint8:
        raise ValueError("Input volume must be uint8")
    if not (0 <= noise_threshold <= 255):
        raise ValueError("noise_threshold must be between 0 and 255")
    if not (1 <= min_votes <= 6):
        raise ValueError("min_votes must be between 1 and 6")

    # Get void mask
    void_mask = process_volume(volume, noise_threshold, min_votes)

    # Apply mask to create output volume
    result = volume.copy()
    result[void_mask == 1] = 0

    return result


@jit(nopython=True)
def get_neighbors_3d(z: int, y: int, x: int, depth: int, height: int, width: int) -> np.ndarray:
    """Get valid 6-connected neighbors for a 3D point."""
    # Pre-allocate maximum possible neighbors
    neighbors = np.zeros((6, 3), dtype=np.int32)
    count = 0

    # Check all 6 directions
    directions = [
        (z - 1, y, x), (z + 1, y, x),  # up/down
        (z, y - 1, x), (z, y + 1, x),  # front/back
        (z, y, x - 1), (z, y, x + 1)  # left/right
    ]

    for nz, ny, nx in directions:
        if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
            neighbors[count, 0] = nz
            neighbors[count, 1] = ny
            neighbors[count, 2] = nx
            count += 1

    return neighbors[:count]


@jit(nopython=True)
def flood_fill_numba(volume: np.ndarray,
                     iso_threshold: int,
                     start_threshold: int) -> np.ndarray:
    """
    Numba-optimized flood fill implementation.
    Returns a mask where 1 indicates kept regions.
    """
    depth, height, width = volume.shape
    mask = np.zeros_like(volume, dtype=np.uint8)
    visited = np.zeros_like(volume, dtype=np.uint8)

    # Pre-allocate queue arrays
    max_queue_size = depth * height * width
    queue_z = np.zeros(max_queue_size, dtype=np.int32)
    queue_y = np.zeros(max_queue_size, dtype=np.int32)
    queue_x = np.zeros(max_queue_size, dtype=np.int32)
    queue_start = 0
    queue_end = 0

    # Find and add starting points
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if volume[z, y, x] >= start_threshold:
                    queue_z[queue_end] = z
                    queue_y[queue_end] = y
                    queue_x[queue_end] = x
                    queue_end += 1
                    mask[z, y, x] = 1
                    visited[z, y, x] = 1

    # Process queue
    while queue_start < queue_end:
        current_z = queue_z[queue_start]
        current_y = queue_y[queue_start]
        current_x = queue_x[queue_start]
        queue_start += 1

        neighbors = get_neighbors_3d(
            current_z, current_y, current_x,
            depth, height, width
        )

        for i in range(len(neighbors)):
            z = neighbors[i, 0]
            y = neighbors[i, 1]
            x = neighbors[i, 2]

            if visited[z, y, x] == 1 or volume[z, y, x] < iso_threshold:
                continue

            mask[z, y, x] = 1
            visited[z, y, x] = 1
            queue_z[queue_end] = z
            queue_y[queue_end] = y
            queue_x[queue_end] = x
            queue_end += 1

    return mask


@jit(nopython=True)
def segment_and_clean(volume: np.ndarray,
                      iso_threshold: int = 30,
                      start_threshold: int = 200) -> np.ndarray:
    """
    Segment and clean a volume using flood fill.
    All Numba-compatible operations.
    """
    depth, height, width = volume.shape
    result = np.zeros_like(volume)

    # Get the flood fill mask
    mask = flood_fill_numba(volume, iso_threshold, start_threshold)

    # Apply mask using explicit loops
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if mask[z, y, x] == 1:
                    result[z, y, x] = volume[z, y, x]
                # else: result stays 0

    return result


@jit(nopython=True)
def flood_fill_segment_numba(volume: np.ndarray,
                             iso_threshold: int,
                             start_threshold: int) -> np.ndarray:
    """
    Numba-optimized flood fill segmentation.
    """
    depth, height, width = volume.shape
    mask = np.zeros_like(volume, dtype=np.uint8)
    visited = np.zeros_like(volume, dtype=np.uint8)

    # Pre-allocate queue arrays (worst case: entire volume)
    # Using separate arrays for z, y, x coordinates
    max_queue_size = depth * height * width
    queue_z = np.zeros(max_queue_size, dtype=np.int32)
    queue_y = np.zeros(max_queue_size, dtype=np.int32)
    queue_x = np.zeros(max_queue_size, dtype=np.int32)
    queue_start = 0  # Front of queue
    queue_end = 0  # Back of queue

    # Find and add starting points
    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if volume[z, y, x] >= start_threshold:
                    queue_z[queue_end] = z
                    queue_y[queue_end] = y
                    queue_x[queue_end] = x
                    queue_end += 1
                    mask[z, y, x] = 1
                    visited[z, y, x] = 1

    # Process queue
    while queue_start < queue_end:
        # Get current point
        current_z = queue_z[queue_start]
        current_y = queue_y[queue_start]
        current_x = queue_x[queue_start]
        queue_start += 1

        # Get neighbors
        neighbors = get_neighbors_3d(
            current_z, current_y, current_x,
            depth, height, width
        )

        # Process each neighbor
        for i in range(len(neighbors)):
            z = neighbors[i, 0]
            y = neighbors[i, 1]
            x = neighbors[i, 2]

            # Skip if already visited or below threshold
            if visited[z, y, x] == 1 or volume[z, y, x] < iso_threshold:
                continue

            # Mark as part of region and add to queue
            mask[z, y, x] = 1
            visited[z, y, x] = 1
            queue_z[queue_end] = z
            queue_y[queue_end] = y
            queue_x[queue_end] = x
            queue_end += 1

    return mask


@jit(nopython=True)
def apply_mask_numba(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to volume using explicit loops for Numba compatibility."""
    result = volume.copy()
    depth, height, width = volume.shape

    for z in range(depth):
        for y in range(height):
            for x in range(width):
                if mask[z, y, x] == 0:
                    result[z, y, x] = 0

    return result

