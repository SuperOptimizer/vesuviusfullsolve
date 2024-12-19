import zarr
import os
import numpy as np

from snic import Superpixel

def superclusters_to_numpy(superclusters, padding=(0, 0, 0)):
    num_points = 64 * 64 * 64  # Maximum points per chunk
    data = np.full((num_points, 5), np.nan, dtype=np.float16)

    for i, sc in enumerate(superclusters):
        if i >= num_points:
            print(f"Warning: More superclusters ({len(superclusters)}) than available slots ({num_points})")
            break

        data[i] = [
            sc.z - padding[0],
            sc.y - padding[1],
            sc.x - padding[2],
            sc.c,
            sc.n
        ]
    return data


def numpy_to_superclusters(data):
    superclusters = []
    for point in data:
        # Skip nan entries
        if np.isnan(point[0]):
            continue

        # Create Superpixel object
        sc = Superpixel(
            z=float(point[0]),
            y=float(point[1]),
            x=float(point[2]),
            c=float(point[3]),
            n=int(point[4])
        )
        superclusters.append(sc)

    return superclusters


def paths_to_numpy(paths, superclusters, max_paths, max_path_length):
    """Convert paths to numpy array for zarr storage.

    Args:
        paths: List of sets of superpixels representing paths
        superclusters: List of all superpixels for this chunk
        max_paths: Maximum number of paths to store
        max_path_length: Maximum length of each path

    Returns:
        numpy array of shape (max_paths, max_path_length) containing superpixel indices,
        with -1 for padding
    """
    # Initialize array with -1 (indicating no superpixel)
    data = np.full((max_paths, max_path_length), -1, dtype=np.int32)

    # Create mapping from superpixel to index
    supercluster_to_idx = {id(sc): idx for idx, sc in enumerate(superclusters)}

    # Fill in path data
    for path_idx, path in enumerate(paths):
        if path_idx >= max_paths:
            print(f"Warning: More paths ({len(paths)}) than available slots ({max_paths})")
            break

        # Convert path set to list of indices
        path_indices = [supercluster_to_idx[id(sp)] for sp in path]

        # Truncate if necessary
        if len(path_indices) > max_path_length:
            print(f"Warning: Path {path_idx} truncated from {len(path_indices)} to {max_path_length}")
            path_indices = path_indices[:max_path_length]

        # Store indices
        data[path_idx, :len(path_indices)] = path_indices

    return data


def numpy_to_paths(data, superclusters):
    """Convert numpy array from zarr storage back to paths.

    Args:
        data: numpy array of shape (max_paths, max_path_length) containing superpixel indices
        superclusters: List of all superpixels for this chunk

    Returns:
        List of sets containing superpixels representing paths
    """
    paths = []

    # Process each non-empty path
    for path_indices in data:
        # Find where path ends (-1 indicates no more superpixels)
        valid_indices = path_indices[path_indices >= 0]

        if len(valid_indices) > 0:
            # Convert indices back to superpixels
            path = {superclusters[idx] for idx in valid_indices}
            paths.append(path)

    return paths
