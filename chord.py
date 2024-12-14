import numpy as np
from scipy.spatial import cKDTree
from typing import Set, Dict, List, Tuple, Optional, NamedTuple
import numpy.typing as npt
from dataclasses import dataclass
#chord.py

@dataclass
class GrowthStats:
    active_chords: int
    avg_length: float
    min_length: int
    max_length: int
    total_points_used: int


@dataclass
class VolumeTracker:
    bounds: np.ndarray
    direction_tree: cKDTree = None
    directions: List[Tuple[np.ndarray, np.ndarray]] = None  # [(pos, dir)]

    @classmethod
    def initialize(cls, bounds: np.ndarray):
        return cls(bounds, None, [])

    def update(self, pos: np.ndarray, direction: np.ndarray):
        self.directions.append((pos, direction))
        if len(self.directions) % 100 == 0:  # Rebuild tree periodically
            positions = np.array([p for p, _ in self.directions])
            self.direction_tree = cKDTree(positions)

    def get_parallel_score(self, pos: np.ndarray, proposed_dir: np.ndarray) -> float:
        if not self.direction_tree:
            return 1.0

        distances, indices = self.direction_tree.query(pos, k=5, distance_upper_bound=5.0)
        valid_indices = indices[distances != np.inf]

        if not len(valid_indices):
            return 1.0

        nearby_dirs = [self.directions[i][1] for i in valid_indices]
        alignments = [abs(np.dot(proposed_dir, d)) for d in nearby_dirs]
        return np.mean(alignments)


def initialize_points(points: List, bounds: npt.NDArray[np.float32]) -> Tuple[np.ndarray, Dict[int, int], np.ndarray]:
    n_points = len(points)
    positions = np.empty((n_points, 3), dtype=np.float32)
    intensities = np.empty(n_points, dtype=np.float32)

    for i, point in enumerate(points):
        positions[i] = [float(point.z), float(point.y), float(point.x)]
        intensities[i] = float(point.c)

    id_to_index = {id(point): i for i, point in enumerate(points)}
    return positions, id_to_index, intensities


def select_start_points(points: List,
                        positions: np.ndarray,
                        intensities: np.ndarray,
                        bounds: npt.NDArray[np.float32],
                        target_count: int = 4096,
                        min_intensity_percentile: int = 30) -> List[int]:
    z_min, z_max = bounds[0, 0], bounds[0, 1]
    n_layers = int(np.sqrt(target_count))
    z_step = (z_max - z_min) / n_layers
    points_per_layer = target_count // n_layers

    starts = []
    min_intensity = np.percentile(intensities, min_intensity_percentile)
    spatial_index = cKDTree(positions)

    for layer in range(n_layers):
        layer_mask = (positions[:, 0] >= z_min + layer * z_step) & \
                     (positions[:, 0] < z_min + (layer + 1) * z_step)
        layer_points = np.where(layer_mask)[0]

        if len(layer_points) == 0:
            continue

        valid_points = [idx for idx in layer_points
                        if intensities[idx] > min_intensity and
                        hasattr(points[idx], 'connections') and
                        len(points[idx].connections) >= 4]

        if not valid_points:
            continue

        selected = []
        while len(selected) < points_per_layer and valid_points:
            idx = np.random.choice(valid_points)
            if selected:
                nearest_dist = min(np.linalg.norm(positions[idx] - positions[s])
                                   for s in selected)
                if nearest_dist < z_step / 2:
                    continue
            selected.append(idx)
            valid_points.remove(idx)

        starts.extend(selected)
        if len(starts) >= target_count:
            break

    return starts


def grow_chord(start_point: int,
               points: List,
               positions: np.ndarray,
               intensities: np.ndarray,
               id_to_index: Dict[int, int],
               available: np.ndarray,
               volume_tracker: VolumeTracker,
               min_length: int = 8,
               max_length: int = 32) -> Optional[Set]:
    current = points[start_point]
    chord = {current}
    pos = positions[start_point]

    for direction in [1, -1]:
        current_pos = pos.copy()
        current_point = current
        recent_dirs = []

        while len(chord) < max_length:
            if not hasattr(current_point, 'connections') or not current_point.connections:
                break

            candidates = []
            for next_point, strength in current_point.connections.items():
                idx = id_to_index[id(next_point)]
                if not available[idx]:
                    continue

                next_pos = positions[idx]
                dp = next_pos - current_pos
                dist = np.linalg.norm(dp)

                if dist < 0.1:
                    continue

                if (direction > 0 and next_pos[0] <= current_pos[0]) or \
                        (direction < 0 and next_pos[0] >= current_pos[0]):
                    continue

                dp_norm = dp / dist

                # Enforce z-direction progress
                z_progress = direction * dp_norm[0]
                if z_progress < 0.5:  # Require significant z movement
                    continue

                # Calculate smoothness using last 3 directions
                smoothness_score = 1.0
                if recent_dirs:
                    smoothness_scores = [np.dot(dp_norm, d) for d in recent_dirs[:3]]
                    smoothness_score = np.mean(smoothness_scores)
                    if smoothness_score < 0.8:  # Strict smoothness requirement
                        continue

                parallel_score = volume_tracker.get_parallel_score(next_pos, dp_norm)

                total_score = (
                        (strength / 255.0) * 0.3 +
                        z_progress * 0.2 +  # Heavily weight z-direction progress
                        smoothness_score * 0.3 +
                        parallel_score * 0.2
                )

                candidates.append((idx, next_pos, total_score, dp_norm))

            if not candidates:
                break

            idx, pos, _, direction_vec = max(candidates, key=lambda x: x[2])
            point = points[idx]

            chord.add(point)
            available[idx] = False
            current_pos = pos
            current_point = point

            recent_dirs = [direction_vec] + recent_dirs[:2]
            volume_tracker.update(current_pos, direction_vec)

    return chord if len(chord) >= min_length else None


def grow_fiber_chords(points: List,
                      bounds: npt.NDArray[np.float32],
                      num_chords: int = 4096,
                      min_length: int = 8,
                      max_length: int = 32) -> List[Set]:
    positions, id_to_index, intensities = initialize_points(points, bounds)
    available = np.ones(len(points), dtype=bool)
    volume_tracker = VolumeTracker.initialize(bounds)

    start_indices = select_start_points(
        points, positions, intensities, bounds,
        target_count=num_chords
    )

    if not start_indices:
        return []

    completed_chords = []
    for i, start_idx in enumerate(start_indices):
        if not available[start_idx]:
            continue

        chord = grow_chord(
            start_idx,
            points,
            positions,
            intensities,
            id_to_index,
            available,
            volume_tracker,
            min_length,
            max_length
        )

        if chord:
            completed_chords.append(chord)

    return completed_chords


def get_growth_stats(chords: List[Set]) -> GrowthStats:
    if not chords:
        return GrowthStats(0, 0.0, 0, 0, 0)

    lengths = [len(c) for c in chords]
    return GrowthStats(
        active_chords=len(chords),
        avg_length=np.mean(lengths),
        min_length=min(lengths),
        max_length=max(lengths),
        total_points_used=sum(lengths)
    )