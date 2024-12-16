import numpy as np
from scipy.spatial import cKDTree
from typing import Set, Dict, List, Tuple, Optional, NamedTuple, Literal
import numpy.typing as npt
from dataclasses import dataclass
import random
from collections import defaultdict


@dataclass
class GrowthStats:
    active_paths: int
    avg_length: float
    min_length: int
    max_length: int
    total_points_used: int


@dataclass
class VolumeTracker:
    bounds: np.ndarray
    direction_tree: cKDTree = None
    directions: List[Tuple[np.ndarray, np.ndarray]] = None

    @classmethod
    def initialize(cls, bounds: np.ndarray):
        return cls(bounds, None, [])

    def update(self, pos: np.ndarray, direction: np.ndarray):
        if self.directions is None:
            self.directions = []
        self.directions.append((pos, direction))
        if len(self.directions) % 100 == 0:
            positions = np.array([p for p, _ in self.directions])
            self.direction_tree = cKDTree(positions)


def initialize_points(points: List, bounds: npt.NDArray[np.float32]) -> Tuple[
    np.ndarray, Dict[int, int], np.ndarray, Dict[int, List[Tuple[int, float]]]]:
    n_points = len(points)
    positions = np.empty((n_points, 3), dtype=np.float32)
    intensities = np.empty(n_points, dtype=np.float32)

    # Pre-compute connections dictionary
    connections_dict = {}
    id_to_index = {}

    for i, point in enumerate(points):
        positions[i] = [float(point.z), float(point.y), float(point.x)]
        intensities[i] = float(point.c)
        id_to_index[id(point)] = i

        if hasattr(point, 'connections'):
            connections = []
            for next_point, strength in point.connections.items():
                next_idx = id_to_index.get(id(next_point))
                if next_idx is not None:
                    connections.append((next_idx, float(strength)))
            connections_dict[i] = connections

    return positions, id_to_index, intensities, connections_dict


import numpy as np
from scipy.spatial import cKDTree
from typing import Set, Dict, List, Tuple, Optional, NamedTuple, Literal
import numpy.typing as npt
from dataclasses import dataclass
import random
from collections import defaultdict
import heapq


@dataclass
class PathSegment:
    points: List[int]  # List of point indices
    positions: List[np.ndarray]  # List of positions
    score: float
    direction: np.ndarray  # Overall direction of segment
    end_direction: np.ndarray  # Direction at end of segment


def find_path_segments(current_idx: int,
                       target_axis: int,
                       look_ahead: int,
                       positions: np.ndarray,
                       connections_dict: Dict[int, List[Tuple[int, float]]],
                       available: np.ndarray,
                       recent_dirs: List[np.ndarray],
                       direction: int) -> List[PathSegment]:
    """Find possible path segments looking ahead N steps"""
    start_pos = positions[current_idx]
    segments = []

    # Use priority queue for breadth-first search of possible paths
    queue = [(0, [current_idx], [start_pos], [])]  # (negative_score, points, positions, directions)
    visited = {current_idx}

    while queue and len(segments) < 10:  # Limit number of segments to consider
        _, path_points, path_positions, path_directions = heapq.heappop(queue)
        current = path_points[-1]

        if len(path_points) >= look_ahead:
            # Calculate segment properties
            total_direction = path_positions[-1] - path_positions[0]
            total_direction = total_direction / np.linalg.norm(total_direction)

            # Calculate overall progress in target direction
            axis_progress = direction * (path_positions[-1][target_axis] - path_positions[0][target_axis])

            # Calculate smoothness
            smoothness = 1.0
            if path_directions:
                direction_dots = [abs(np.dot(d1, d2)) for d1, d2 in zip(path_directions[:-1], path_directions[1:])]
                smoothness = np.mean(direction_dots)

            # Score the segment
            axis_alignment = total_direction[target_axis] * direction
            score = (
                    axis_progress * 0.4 +  # Progress along axis
                    smoothness * 0.3 +  # Path smoothness
                    axis_alignment * 0.3  # Overall direction alignment
            )

            segments.append(PathSegment(
                points=path_points,
                positions=path_positions,
                score=score,
                direction=total_direction,
                end_direction=path_directions[-1] if path_directions else total_direction
            ))
            continue

        # Explore next possible points
        if current not in connections_dict:
            continue

        for next_idx, strength in connections_dict[current]:
            if not available[next_idx] or next_idx in visited:
                continue

            next_pos = positions[next_idx]
            dp = next_pos - path_positions[-1]
            dist = np.linalg.norm(dp)

            if dist < 0.1:
                continue

            dp_norm = dp / dist

            # Calculate temporary score for prioritizing exploration
            temp_score = 0.0

            # Consider alignment with previous direction
            if path_directions:
                dir_alignment = abs(np.dot(dp_norm, path_directions[-1]))
                temp_score -= dir_alignment

            # Consider progress in target direction
            axis_progress = direction * dp_norm[target_axis]
            temp_score -= axis_progress

            # Add to queue
            new_points = path_points + [next_idx]
            new_positions = path_positions + [next_pos]
            new_directions = path_directions + [dp_norm]

            heapq.heappush(queue, (temp_score, new_points, new_positions, new_directions))
            visited.add(next_idx)

    return segments


def grow_path(start_point: int,
              points: List,
              positions: np.ndarray,
              connections_dict: Dict[int, List[Tuple[int, float]]],
              available: np.ndarray,
              volume_tracker: VolumeTracker,
              min_length: int = 8,
              axis: int = 0,
              look_ahead: int = 5,
              point_set: Optional[Set] = None) -> Optional[Set]:
    path = {points[start_point]} if point_set is None else point_set
    current_idx = start_point
    available[current_idx] = False
    path_indices = {current_idx}
    current_pos = positions[current_idx]
    recent_dirs = []
    path_length = 1

    for direction in [1, -1]:
        current_idx = start_point
        current_pos = positions[current_idx]

        while True:
            # Find possible path segments looking ahead
            segments = find_path_segments(
                current_idx,
                axis,
                look_ahead,
                positions,
                connections_dict,
                available,
                recent_dirs,
                direction
            )

            if not segments:
                break

            # Choose best segment (with some randomness)
            top_segments = sorted(segments, key=lambda s: s.score, reverse=True)[:3]
            chosen_segment = random.choice(top_segments)

            # Add segment to path
            for idx in chosen_segment.points[1:]:  # Skip first point as it's already in path
                if idx in path_indices:
                    break

                path.add(points[idx])
                path_indices.add(idx)
                path_length += 1
                available[idx] = False

            # Update state
            current_idx = chosen_segment.points[-1]
            current_pos = positions[current_idx]
            recent_dirs = [chosen_segment.end_direction] + recent_dirs[:2]

            # Update volume tracker with segment direction
            volume_tracker.update(current_pos, chosen_segment.direction)

    return path if path_length >= min_length else None


def select_start_points_fast(positions: np.ndarray,
                             connections_dict: Dict[int, List[Tuple[int, float]]],
                             target_count: int = 256) -> List[int]:
    """Faster start point selection using numpy operations"""
    valid_points = np.array(list(connections_dict.keys()))
    if len(valid_points) == 0:
        return []

    # Use numpy's random choice for initial selection
    if len(valid_points) <= target_count:
        return valid_points.tolist()

    selected_indices = np.random.choice(len(valid_points),
                                        size=min(target_count * 2, len(valid_points)),
                                        replace=False)
    return valid_points[selected_indices].tolist()


def grow_fiber_paths(points: List,
                     bounds: npt.NDArray[np.float32],
                     num_paths: int = 512,
                     min_length: int = 8) -> Tuple[List[Set], List[Set], List[Set]]:
    positions, id_to_index, intensities, connections_dict = initialize_points(points, bounds)
    paths_per_direction = num_paths // 3

    all_paths = []
    for axis in range(3):
        print(f"\nProcessing axis {axis}")
        available = np.ones(len(points), dtype=bool)
        volume_tracker = VolumeTracker.initialize(bounds)

        start_indices = select_start_points_fast(
            positions,
            connections_dict,
            target_count=paths_per_direction * 2
        )

        axis_paths = []
        for start_idx in start_indices:
            if not available[start_idx]:
                continue

            path = grow_path(
                start_idx,
                points,
                positions,
                connections_dict,
                available,
                volume_tracker,
                min_length=min_length,
                axis=axis
            )

            if path:
                axis_paths.append(path)
                if len(axis_paths) >= paths_per_direction:
                    break

        all_paths.append(axis_paths)
        print(f"Generated {len(axis_paths)} valid paths for axis {axis}")

    return all_paths[0], all_paths[1], all_paths[2]