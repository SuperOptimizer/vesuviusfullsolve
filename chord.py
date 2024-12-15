import numpy as np
from scipy.spatial import cKDTree
from typing import Set, Dict, List, Tuple, Optional, NamedTuple, Literal
import numpy.typing as npt
from dataclasses import dataclass


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
                        n_layers: int = 256,
                        min_intensity_percentile: int = 5,
                        axis: int = 0) -> List[int]:
    axis_min, axis_max = bounds[axis, 0], bounds[axis, 1]
    axis_step = (axis_max - axis_min) / n_layers
    points_per_layer = target_count // n_layers

    starts = []
    min_intensity = np.percentile(intensities, min_intensity_percentile)

    for layer in range(n_layers):
        layer_mask = (positions[:, axis] >= axis_min + layer * axis_step) & \
                     (positions[:, axis] < axis_min + (layer + 1) * axis_step)
        layer_points = np.where(layer_mask)[0]

        if len(layer_points) == 0:
            continue

        valid_points = [idx for idx in layer_points
                        if intensities[idx] > min_intensity and
                        hasattr(points[idx], 'connections') and
                        len(points[idx].connections) >= 4]

        if not valid_points:
            continue

        selected = np.random.choice(valid_points,
                                    size=min(points_per_layer, len(valid_points)),
                                    replace=False)
        starts.extend(selected)

    return starts


def get_chord_direction(positions: np.ndarray, idx: int, window: int = 3) -> np.ndarray:
    start = max(0, idx - window)
    end = min(len(positions), idx + window + 1)
    if end - start < 2:
        return np.array([1, 0, 0])  # Default to z-direction

    points = positions[start:end]
    direction = points[-1] - points[0]
    return direction / np.linalg.norm(direction)


def grow_chord(start_point: int,
               points: List,
               positions: np.ndarray,
               intensities: np.ndarray,
               id_to_index: Dict[int, int],
               available: np.ndarray,
               volume_tracker: VolumeTracker,
               min_length: int = 8,
               max_length: int = 32,
               axis: int = 0) -> Optional[Set]:
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

                if dist < .1:
                    continue

                if (direction > 0 and next_pos[axis] <= current_pos[axis]) or \
                        (direction < 0 and next_pos[axis] >= current_pos[axis]):
                    continue

                dp_norm = dp / dist

                axis_progress = direction * dp_norm[axis]
                if axis_progress < 0.5:
                    continue

                smoothness_score = 1.0
                if recent_dirs:
                    smoothness_scores = [np.dot(dp_norm, d) for d in recent_dirs[:3]]
                    smoothness_score = np.mean(smoothness_scores)
                    if smoothness_score < 0.8:
                        continue

                parallel_score = volume_tracker.get_parallel_score(next_pos, dp_norm)

                total_score = (
                    (strength / 255.0) * 0.4 +
                    axis_progress * 0.2 +
                    parallel_score * 0.4
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
                      max_length: int = 32) -> Tuple[List[Set], List[Set], List[Set]]:
    positions, id_to_index, intensities = initialize_points(points, bounds)
    chords_per_direction = num_chords // 3

    z_chords = []
    y_chords = []
    x_chords = []

    # Grow Z direction chords
    available = np.ones(len(points), dtype=bool)
    volume_tracker = VolumeTracker.initialize(bounds)
    start_indices = select_start_points(points, positions, intensities, bounds,
                                        target_count=chords_per_direction)

    for start_idx in start_indices:
        if not available[start_idx]:
            continue
        chord = grow_chord(start_idx, points, positions, intensities,
                           id_to_index, available, volume_tracker,
                           min_length=min_length, max_length=max_length)
        if chord:
            z_chords.append(chord)

    # Grow Y direction chords
    available = np.ones(len(points), dtype=bool)
    volume_tracker = VolumeTracker.initialize(bounds)
    start_indices = select_start_points(points, positions, intensities, bounds,
                                        target_count=chords_per_direction)

    for start_idx in start_indices:
        if not available[start_idx]:
            continue
        chord = grow_chord(start_idx, points, positions, intensities,
                           id_to_index, available, volume_tracker,
                           axis=1, min_length=min_length, max_length=max_length)
        if chord:
            y_chords.append(chord)

    # Grow X direction chords
    available = np.ones(len(points), dtype=bool)
    volume_tracker = VolumeTracker.initialize(bounds)
    start_indices = select_start_points(points, positions, intensities, bounds,
                                        target_count=chords_per_direction)

    for start_idx in start_indices:
        if not available[start_idx]:
            continue
        chord = grow_chord(start_idx, points, positions, intensities,
                           id_to_index, available, volume_tracker,
                           axis=2, min_length=min_length, max_length=max_length)
        if chord:
            x_chords.append(chord)

    return z_chords, y_chords, x_chords


def find_chord_joins(chords: List[Set],
                     positions: np.ndarray,
                     id_to_index: Dict[int, int],
                     max_distance: float = 2.0,
                     min_parallel_score: float = 0.8) -> List[Tuple[Set, Set, int, int]]:

    joins = []
    all_positions = []
    point_to_chord = {}

    for chord_idx, chord in enumerate(chords):
        for point in chord:
            idx = id_to_index[id(point)]
            pos = positions[idx]
            all_positions.append(pos)
            point_to_chord[tuple(pos)] = (chord_idx, point)

    tree = cKDTree(all_positions)
    pairs = tree.query_pairs(max_distance)

    for pos1_idx, pos2_idx in pairs:
        pos1 = all_positions[pos1_idx]
        pos2 = all_positions[pos2_idx]

        chord1_idx, point1 = point_to_chord[tuple(pos1)]
        chord2_idx, point2 = point_to_chord[tuple(pos2)]

        if chord1_idx >= chord2_idx:
            continue

        chord1 = chords[chord1_idx]
        chord2 = chords[chord2_idx]

        dir1 = get_chord_direction(
            np.array([positions[id_to_index[id(p)]] for p in chord1]),
            list(chord1).index(point1)
        )
        dir2 = get_chord_direction(
            np.array([positions[id_to_index[id(p)]] for p in chord2]),
            list(chord2).index(point2)
        )

        if abs(np.dot(dir1, dir2)) > min_parallel_score:
            joins.append((
                chord1, chord2,
                list(chord1).index(point1),
                list(chord2).index(point2)
            ))

    return joins


def would_create_cycle(chord1: Set, chord2: Set, idx1: int, idx2: int) -> bool:
    return bool(chord1.intersection(chord2))


def merge_chords(chord1: Set, chord2: Set, idx1: int, idx2: int) -> Set:
    points1 = list(chord1)
    points2 = list(chord2)
    point1, point2 = points1[idx1], points2[idx2]
    z1, z2 = float(point1.z), float(point2.z)

    if z1 < z2:
        merged = set(points1[:idx1 + 1]) | set(points2[idx2:])
    else:
        merged = set(points2[:idx2 + 1]) | set(points1[idx1:])

    return merged


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


def analyze_chord_lengths(patches: List[List[Set]],
                          z_chords: List[Set],
                          y_chords: List[Set],
                          x_chords: List[Set],
                          positions: np.ndarray,
                          id_to_index: Dict[int, int]) -> Dict:
    def get_chord_length(chord: Set) -> float:
        points = np.array([positions[id_to_index[id(p)]] for p in chord])
        return np.linalg.norm(points[-1] - points[0])

    # Calculate lengths for all chords
    z_lengths = [get_chord_length(c) for c in z_chords]
    y_lengths = [get_chord_length(c) for c in y_chords]
    x_lengths = [get_chord_length(c) for c in x_chords]

    # Analyze which chords ended up in patches
    used_chords = set()
    for patch in patches:
        for chord in patch:
            used_chords.add(id(chord))

    z_used = [c for c in z_chords if id(c) in used_chords]
    y_used = [c for c in y_chords if id(c) in used_chords]
    x_used = [c for c in x_chords if id(c) in used_chords]

    z_used_lengths = [get_chord_length(c) for c in z_used]
    y_used_lengths = [get_chord_length(c) for c in y_used]
    x_used_lengths = [get_chord_length(c) for c in x_used]

    return {
        'z_stats': {
            'total': len(z_chords),
            'used': len(z_used),
            'avg_length': np.mean(z_lengths),
            'used_avg_length': np.mean(z_used_lengths) if z_used_lengths else 0,
            'length_hist': np.histogram(z_lengths, bins=10)[0].tolist()
        },
        'y_stats': {
            'total': len(y_chords),
            'used': len(y_used),
            'avg_length': np.mean(y_lengths),
            'used_avg_length': np.mean(y_used_lengths) if y_used_lengths else 0,
            'length_hist': np.histogram(y_lengths, bins=10)[0].tolist()
        },
        'x_stats': {
            'total': len(x_chords),
            'used': len(x_used),
            'avg_length': np.mean(x_lengths),
            'used_avg_length': np.mean(x_used_lengths) if x_used_lengths else 0,
            'length_hist': np.histogram(x_lengths, bins=10)[0].tolist()
        }
    }


def print_length_analysis(stats: Dict):
    for dir_name, dir_stats in stats.items():
        print(f"\n{dir_name}:")
        print(f"Total chords: {dir_stats['total']}")
        print(f"Used in patches: {dir_stats['used']} ({dir_stats['used'] / dir_stats['total'] * 100:.1f}%)")
        print(f"Average length: {dir_stats['avg_length']:.2f}")
        print(f"Average length of used chords: {dir_stats['used_avg_length']:.2f}")
        print("Length distribution:", dir_stats['length_hist'])