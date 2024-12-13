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


import numpy as np
from numba import jit
import numpy.typing as npt


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


def get_chunk_slices(shape, chunk_size):
    """
    Generate slices for chunking a 3D volume.

    Parameters:
    -----------
    shape : tuple
        Shape of the volume (depth, height, width)
    chunk_size : int
        Size of cubic chunks

    Returns:
    --------
    list of tuples
        Each tuple contains (depth_slice, height_slice, width_slice)
    """
    depth, height, width = shape
    slices = []

    for d in range(0, depth, chunk_size):
        for h in range(0, height, chunk_size):
            for w in range(0, width, chunk_size):
                d_slice = slice(d, min(d + chunk_size, depth))
                h_slice = slice(h, min(h + chunk_size, height))
                w_slice = slice(w, min(w + chunk_size, width))
                slices.append((d_slice, h_slice, w_slice))

    return slices


def compute_global_params(volume, chunk_size=256, l=256):
    """
    Compute global parameters from a volume by processing it in chunks,
    ignoring zero values.

    Parameters:
    -----------
    volume : ndarray
        Input 3D volume as uint8
    chunk_size : int
        Size of cubic chunks
    l : int
        Number of intensity levels (default 256 for uint8)

    Returns:
    --------
    dict containing:
        - lambda_opt: optimal lambda value
        - h_u: uniform histogram
        - total_size: total number of non-zero voxels
    """
    if volume.dtype != np.uint8:
        raise ValueError("Volume must be uint8")

    # Initialize histogram (index 0 will be ignored)
    h_i = np.zeros(l, dtype=np.float32)
    total_nonzero = 0

    # Get chunk slices
    chunk_slices = get_chunk_slices(volume.shape, chunk_size)

    # Process each chunk
    for d_slice, h_slice, w_slice in chunk_slices:
        chunk = volume[d_slice, h_slice, w_slice]
        # Only consider non-zero values
        nonzero_values = chunk[chunk > 0]
        if nonzero_values.size > 0:
            chunk_hist = np.bincount(nonzero_values.flatten(), minlength=l)
            h_i += chunk_hist
            total_nonzero += nonzero_values.size

    # Set zero-value bin to 0 and normalize histogram using only non-zero counts
    h_i[0] = 0
    h_i = h_i.astype(np.float32) / total_nonzero if total_nonzero > 0 else h_i

    # Create uniform histogram excluding zero
    h_u = np.ones(l, dtype=np.float32)
    h_u[0] = 0  # Zero bin is excluded
    h_u = h_u / (l - 1)  # Normalize excluding zero bin

    # Optimize lambda
    result = minimize_scalar(f3d, method="brent", args=(h_i, h_u, l))

    return {
        'lambda_opt': result.x,
        'h_u': h_u,
        'total_nonzero': total_nonzero
    }


def process_chunk(chunk, global_params, l=256):
    """
    Process a single chunk using pre-computed global parameters,
    preserving zero values.

    Parameters:
    -----------
    chunk : ndarray
        3D volume chunk as uint8
    global_params : dict
        Dictionary containing global parameters from compute_global_params
    l : int
        Number of intensity levels (default 256 for uint8)

    Returns:
    --------
    output : ndarray
        Contrast-enhanced chunk as uint8
    """
    if chunk.dtype != np.uint8:
        raise ValueError("Chunk must be uint8")

    # Create mask for non-zero values
    nonzero_mask = chunk > 0

    # Handle NaN and infinity values
    chunk = np.nan_to_num(chunk, nan=0, posinf=255, neginf=0).astype(np.uint8)

    # Calculate local histogram only for non-zero values
    nonzero_values = chunk[nonzero_mask]
    if nonzero_values.size > 0:
        h_i_local = np.bincount(nonzero_values.flatten(), minlength=l)
        h_i_local = h_i_local.astype(np.float32) / nonzero_values.size
    else:
        h_i_local = np.zeros(l, dtype=np.float32)

    # Set zero bin to 0
    h_i_local[0] = 0

    # Use global parameters
    lam = global_params['lambda_opt']
    h_u = global_params['h_u']

    # Calculate mapping
    h_tilde = 1 / (1 + lam) * h_i_local + lam / (1 + lam) * h_u
    t = mapping3d(h_tilde, l)

    # Force mapping of 0 to remain 0
    t[0] = 0

    # Create output array
    output = np.zeros_like(chunk)
    output[nonzero_mask] = t[chunk[nonzero_mask]]

    return output


def apply_chunked_glcae_3d(volume, chunk_size=256, l=256):
    """
    Apply GLCAE to a large 3D volume using chunking.

    Parameters:
    -----------
    volume : ndarray
        Input 3D volume as uint8
    chunk_size : int
        Size of cubic chunks (default 256)
    l : int
        Number of intensity levels (default 256 for uint8)

    Returns:
    --------
    output : ndarray
        Contrast-enhanced volume as uint8
    """
    if volume.dtype != np.uint8:
        raise ValueError("Input volume must be uint8")

    # Initialize output volume
    output = np.zeros_like(volume)

    # Compute global parameters
    global_params = compute_global_params(volume, chunk_size, l)

    # Get chunk slices
    chunk_slices = get_chunk_slices(volume.shape, chunk_size)

    # Process each chunk
    for d_slice, h_slice, w_slice in chunk_slices:
        # Extract and process chunk
        chunk = volume[d_slice, h_slice, w_slice]
        enhanced_chunk = process_chunk(chunk, global_params, l)

        # Store result
        output[d_slice, h_slice, w_slice] = enhanced_chunk

    return output

