import numpy as np
from scipy.spatial import cKDTree
from typing import Set, Dict, List, Tuple, Optional, NamedTuple, Literal
import numpy.typing as npt
from dataclasses import dataclass
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numba
from numba import jit, prange

MAX_PATHS=8192
MIN_PATH_LENGTH=8
MAX_PATH_LENGTH=128

'''

    bounding_box = np.array([
        [0, dims[0]],
        [0, dims[1]],
        [0, dims[2]],
    ])

    zpaths = path.grow_paths_parallel(
        points=superclusters,
        bounds=bounding_box,
        axis=0,
        num_paths=4096,  # Reduced number for longer paths
        min_length=8,  # Increased minimum length
        max_length=256
    )

    return vis.visualize_paths(zpaths,[],[])

'''

@dataclass
class ActivePath:
    points: np.ndarray  # Array of point indices
    point_count: int
    front_idx: int
    back_idx: int
    axis: int
    front_pos: np.ndarray
    back_pos: np.ndarray
    front_dir: np.ndarray
    back_dir: np.ndarray
    grow_direction: int  # 1 for positive direction, -1 for negative direction
    is_active: bool = True


@jit(nopython=True)
def is_near_boundary(pos: np.ndarray, min_bounds: np.ndarray, max_bounds: np.ndarray) -> bool:
    return np.any(pos < min_bounds) or np.any(pos > max_bounds)


@jit(nopython=True)
def get_zero_direction():
    """Return a zero direction vector with consistent type"""
    return np.zeros(3, dtype=np.float32)

def initialize_points(points: List, bounds: npt.NDArray[np.float32]) -> Tuple[
    np.ndarray, Dict[int, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize points with flat arrays"""
    n_points = len(points)
    positions = np.empty((n_points, 3), dtype=np.float32)
    intensities = np.empty(n_points, dtype=np.float32)
    id_to_index = {}

    for i, point in enumerate(points):
        positions[i] = [float(point.z), float(point.y), float(point.x)]
        intensities[i] = float(point.c)
        id_to_index[id(point)] = i

    max_connections = 0
    for point in points:
        if hasattr(point, 'connections'):
            max_connections = max(max_connections, len(point.connections))

    connection_indices = np.full((n_points, max_connections), -1, dtype=np.int32)
    connection_strengths = np.zeros((n_points, max_connections), dtype=np.float32)
    connection_counts = np.zeros(n_points, dtype=np.int32)

    for i, point in enumerate(points):
        if hasattr(point, 'connections'):
            for j, (next_point, strength) in enumerate(point.connections.items()):
                next_idx = id_to_index.get(id(next_point))
                if next_idx is not None:
                    connection_indices[i, j] = next_idx
                    connection_strengths[i, j] = float(strength)
                    connection_counts[i] = j + 1

    return (positions, id_to_index, intensities,
            connection_indices, connection_strengths, connection_counts)



@jit(nopython=True)
def find_best_next_point_numba(current_idx: int,
                               current_pos: np.ndarray,
                               current_dir: np.ndarray,
                               target_axis: int,
                               direction: int,
                               positions: np.ndarray,
                               connection_indices: np.ndarray,
                               connection_strengths: np.ndarray,
                               connection_counts: np.ndarray,
                               available: np.ndarray,
                               min_bounds: np.ndarray,
                               max_bounds: np.ndarray) -> Tuple[np.int32, np.ndarray]:
    """Next point selection with strong directional bias"""
    count = connection_counts[current_idx]
    zero_dir = get_zero_direction()

    if count == 0:
        return np.int32(-1), zero_dir

    # Get ideal direction based on target axis
    ideal_dir = np.zeros(3, dtype=np.float32)
    ideal_dir[target_axis] = float(direction)

    curr_connections = connection_indices[current_idx, :count]
    curr_strengths = connection_strengths[current_idx, :count]

    best_score = np.float32(-np.inf)
    best_idx = np.int32(-1)
    best_direction = zero_dir

    for i in range(len(curr_connections)):
        next_idx = curr_connections[i]
        if next_idx < 0 or not available[next_idx]:
            continue

        next_pos = positions[next_idx]
        if is_near_boundary(next_pos, min_bounds, max_bounds):
            continue

        dp = next_pos - current_pos
        dist = np.sqrt(np.sum(dp * dp))

        if dist < 0.1:
            continue

        dp_norm = dp / dist

        # Calculate progress in target direction
        progress = dp_norm[target_axis] * direction

        # Skip points that don't make enough progress
        if progress < 0.3:  # Require at least 30% progress in target direction
            continue

        # Calculate alignment with ideal direction
        alignment_score = np.dot(dp_norm, ideal_dir)

        # Strong preference for directional growth
        score = alignment_score

        if score > best_score:
            best_score = score
            best_idx = np.int32(next_idx)
            best_direction = dp_norm.astype(np.float32)

    return best_idx, best_direction


def grow_path_segment(path: ActivePath,
                      positions: np.ndarray,
                      connection_indices: np.ndarray,
                      connection_strengths: np.ndarray,
                      connection_counts: np.ndarray,
                      available: np.ndarray,
                      min_bounds: np.ndarray,
                      max_bounds: np.ndarray,
                      max_length: int) -> bool:
    """Grow path with strong directional bias, maintaining spatial ordering"""
    if not path.is_active or path.point_count >= max_length:
        path.is_active = False
        return False

    # Check if we're still making good progress overall
    if path.point_count > 1:
        total_dp = positions[path.front_idx] - positions[path.points[0]]
        total_dist = np.sqrt(np.sum(total_dp * total_dp))
        if total_dist > 0:
            overall_progress = (total_dp[path.axis] * path.grow_direction) / total_dist
            if overall_progress < 0.5:  # Stop if overall progress is too low
                path.is_active = False
                return False

    next_idx, next_dir = find_best_next_point_numba(
        path.front_idx,
        path.front_pos,
        path.front_dir,
        path.axis,
        path.grow_direction,
        positions,
        connection_indices,
        connection_strengths,
        connection_counts,
        available,
        min_bounds,
        max_bounds
    )

    if next_idx >= 0:
        if path.point_count >= len(path.points):
            new_points = np.empty(min(len(path.points) * 2, max_length), dtype=np.int32)
            if path.grow_direction > 0:
                # For positive direction, copy existing points to start
                new_points[:path.point_count] = path.points[:path.point_count]
            else:
                # For negative direction, copy existing points to end of new array
                new_start = len(new_points) - path.point_count
                new_points[new_start:] = path.points[:path.point_count]
            path.points = new_points

        if path.grow_direction > 0:
            # Growing towards positive infinity - append
            path.points[path.point_count] = next_idx
        else:
            # Growing towards negative infinity - prepend by shifting
            # First, shift existing points right by 1
            for i in range(path.point_count - 1, -1, -1):
                path.points[i + 1] = path.points[i]
            # Then add new point at beginning
            path.points[0] = next_idx

        path.point_count += 1
        available[next_idx] = False
        next_pos = positions[next_idx]

        path.front_idx = next_idx
        path.front_pos = next_pos
        path.front_dir = next_dir
        return True

    path.is_active = False
    return False

def grow_paths_parallel(points: List,
                       bounds: npt.NDArray[np.float32],
                       axis: int,  # New parameter: 0 for z, 1 for y, 2 for x
                       num_paths: int = 512,
                       min_length: int = 8,
                       max_length: int = 100) -> List[Set]:
    """Volume-based seeding with strong directional growth along a single specified axis"""
    (positions, id_to_index, intensities,
     connection_indices, connection_strengths, connection_counts) = initialize_points(points, bounds)

    paths_per_direction = num_paths // 2  # Divide paths between positive and negative directions
    available = np.ones(len(points), dtype=bool)
    min_bounds = bounds[:, 0] + 0.1
    max_bounds = bounds[:, 1] - 0.2

    # Initialize active paths for the specified axis
    active_paths: List[ActivePath] = []

    # Volume-based seeding for the specified axis
    valid_points = np.array(list(range(len(points))))

    # Create mask for points well within bounds
    volume_mask = ((positions[:, 0] >= min_bounds[0] + 0.2) &
                   (positions[:, 0] <= max_bounds[0] - 0.2) &
                   (positions[:, 1] >= min_bounds[1] + 0.2) &
                   (positions[:, 1] <= max_bounds[1] - 0.2) &
                   (positions[:, 2] >= min_bounds[2] + 0.2) &
                   (positions[:, 2] <= max_bounds[2] - 0.2))

    valid_volume_points = valid_points[volume_mask]

    if len(valid_volume_points) > 0:
        # Seed points for positive direction
        pos_seed_indices = np.random.choice(
            valid_volume_points,
            size=min(paths_per_direction, len(valid_volume_points)),
            replace=False
        )

        for seed_idx in pos_seed_indices:
            if not available[seed_idx]:
                continue

            seed_pos = positions[seed_idx]
            ideal_dir = np.zeros(3, dtype=np.float32)
            ideal_dir[axis] = 1.0

            path = ActivePath(
                points=np.empty(100, dtype=np.int32),
                point_count=1,
                front_idx=seed_idx,
                back_idx=seed_idx,
                axis=axis,
                front_pos=seed_pos,
                back_pos=seed_pos,
                front_dir=ideal_dir,
                back_dir=ideal_dir,
                grow_direction=1  # Positive direction
            )
            path.points[0] = seed_idx
            active_paths.append(path)
            available[seed_idx] = False

        # Seed points for negative direction
        neg_seed_indices = np.random.choice(
            valid_volume_points[available[valid_volume_points]],  # Only choose from still-available points
            size=min(paths_per_direction, np.sum(available[valid_volume_points])),
            replace=False
        )

        for seed_idx in neg_seed_indices:
            if not available[seed_idx]:
                continue

            seed_pos = positions[seed_idx]
            ideal_dir = np.zeros(3, dtype=np.float32)
            ideal_dir[axis] = -1.0

            path = ActivePath(
                points=np.empty(100, dtype=np.int32),
                point_count=1,
                front_idx=seed_idx,
                back_idx=seed_idx,
                axis=axis,
                front_pos=seed_pos,
                back_pos=seed_pos,
                front_dir=ideal_dir,
                back_dir=ideal_dir,
                grow_direction=-1  # Negative direction
            )
            path.points[0] = seed_idx
            active_paths.append(path)
            available[seed_idx] = False

    # Growth phase
    with ThreadPoolExecutor() as executor:
        while True:
            futures = []
            for path in active_paths:
                if path.is_active:
                    futures.append(executor.submit(
                        grow_path_segment,
                        path,
                        positions,
                        connection_indices,
                        connection_strengths,
                        connection_counts,
                        available,
                        min_bounds,
                        max_bounds,
                        max_length
                    ))

            if not futures:
                break

            any_growth = False
            for future in futures:
                any_growth |= future.result()

            if not any_growth:
                break

    # Convert to final format
    final_paths = []
    for path in active_paths:
        if min_length <= path.point_count <= max_length:
            point_set = {points[idx] for idx in path.points[:path.point_count]}
            final_paths.append(point_set)

    return final_paths