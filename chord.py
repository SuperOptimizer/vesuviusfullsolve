import numpy as np
from scipy.spatial import cKDTree
from typing import Set, Dict, List, Tuple, Optional, NamedTuple
import numpy.typing as npt
from dataclasses import dataclass


@dataclass
class GrowthStats:
    """Statistics for monitoring chord growth."""
    active_chords: int
    avg_length: float
    min_length: int
    max_length: int
    total_points_used: int


class ChordState(NamedTuple):
    """Represents a growing fiber-like chord."""
    points: Set  # Set of points in the chord
    head: np.ndarray  # Current position
    direction: np.ndarray  # Current direction
    intensity: float  # Current intensity


def initialize_points(points: List, bounds: npt.NDArray[np.float32]) -> Tuple[
    np.ndarray, Dict[int, int], np.ndarray]:
    """Initialize point data structures with debugging."""
    n_points = len(points)
    positions = np.empty((n_points, 3), dtype=np.float32)
    intensities = np.empty(n_points, dtype=np.float32)

    print(f"\nInitializing {n_points} points")
    print(f"Bounds: {bounds}")

    # Convert points to numpy arrays for faster processing
    for i, point in enumerate(points):
        positions[i] = [float(point.z), float(point.y), float(point.x)]
        intensities[i] = float(point.c)

    id_to_index = {id(point): i for i, point in enumerate(points)}

    # Print statistics
    print(f"Intensity stats - Min: {np.min(intensities):.1f}, Max: {np.max(intensities):.1f}")
    print(f"Mean: {np.mean(intensities):.1f}, Median: {np.median(intensities):.1f}")

    return positions, id_to_index, intensities


def select_start_points(points: List,
                        positions: np.ndarray,
                        intensities: np.ndarray,
                        bounds: npt.NDArray[np.float32],
                        target_count: int = 4096,
                        min_intensity_percentile: int = 30) -> List[int]:
    """Select starting points using a more physically-aware approach."""

    # Calculate z-layers for more uniform distribution
    z_min, z_max = bounds[0, 0], bounds[0, 1]
    n_layers = int(np.sqrt(target_count))
    z_step = (z_max - z_min) / n_layers

    # Points per layer
    points_per_layer = target_count // n_layers

    starts = []
    min_intensity = np.percentile(intensities, min_intensity_percentile)

    print(f"\nSelecting start points:")
    print(f"Number of layers: {n_layers}")
    print(f"Points per layer: {points_per_layer}")
    print(f"Minimum intensity threshold: {min_intensity:.1f}")

    # Build spatial index
    spatial_index = cKDTree(positions)

    # Select points layer by layer
    for layer in range(n_layers):
        z_center = z_min + (layer + 0.5) * z_step

        # Find points in this layer
        layer_mask = (positions[:, 0] >= z_min + layer * z_step) & \
                     (positions[:, 0] < z_min + (layer + 1) * z_step)
        layer_points = np.where(layer_mask)[0]

        if len(layer_points) == 0:
            continue

        # Filter by intensity and connections
        valid_points = []
        for idx in layer_points:
            if (intensities[idx] > min_intensity and
                    hasattr(points[idx], 'connections') and
                    len(points[idx].connections) >= 4):
                valid_points.append(idx)

        if not valid_points:
            continue

        # Randomly select points with some spatial distribution
        selected = []
        while len(selected) < points_per_layer and valid_points:
            idx = np.random.choice(valid_points)

            # Check if point is far enough from already selected points
            if selected:
                nearest_dist = min(np.linalg.norm(positions[idx] - positions[s])
                                   for s in selected)
                if nearest_dist < z_step / 2:  # Minimum spacing
                    continue

            selected.append(idx)
            valid_points.remove(idx)

        starts.extend(selected)

        if len(starts) >= target_count:
            break

    print(f"Selected {len(starts)} start points")
    return starts

def grow_chord(start_point: int,
              points: List,
              positions: np.ndarray,
              intensities: np.ndarray,
              id_to_index: Dict[int, int],
              available: np.ndarray,
              min_length: int = 8,
              max_length: int = 32) -> Optional[Set]:
   """Grow a single chord bidirectionally following strong connections."""
   current = points[start_point]
   chord = {current}
   pos = positions[start_point]
   recent_dirs = []

   for direction in [1, -1]:  # Forward and backward along z
       current_pos = pos.copy()
       current_point = current
       recent_dirs = []  # Reset for each direction

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

               # Basic distance check
               dist = np.linalg.norm(dp)
               if dist < 0.1:
                   continue

               # Enforce z-direction constraint
               if (direction > 0 and next_pos[0] <= current_pos[0]) or \
                  (direction < 0 and next_pos[0] >= current_pos[0]):
                   continue

               dp_norm = dp / dist
               z_score = dp_norm[0] * direction
               if z_score < 0.1:  # Minimum z-alignment
                   continue

               # Connection strength is primary score
               conn_score = strength / 255.0

               # Path smoothness
               smoothness_score = 1.0
               if recent_dirs:
                   avg_dir = np.mean(recent_dirs, axis=0)
                   smoothness_score = np.dot(dp_norm, avg_dir)
                   smoothness_score = (smoothness_score + 1) / 2

               total_score = (
                   conn_score * 0.6 +         # Primary: follow strong connections
                   z_score * 0.2 +            # Secondary: maintain z direction
                   smoothness_score * 0.2     # Tertiary: avoid sharp turns
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

           recent_dirs.append(direction_vec)
           if len(recent_dirs) > 3:
               recent_dirs.pop(0)

   return chord if len(chord) >= min_length else None


def grow_fiber_chords(points: List,
                      bounds: npt.NDArray[np.float32],
                      num_chords: int = 4096,
                      min_length: int = 8,
                      max_length: int = 32) -> List[Set]:
    """Generate fiber-like chords using simplified physical model."""

    # Initialize data structures
    positions, id_to_index, intensities = initialize_points(points, bounds)
    available = np.ones(len(points), dtype=bool)

    # Select starting points
    start_indices = select_start_points(
        points, positions, intensities, bounds,
        target_count=num_chords
    )

    if not start_indices:
        print("No valid starting points found!")
        return []

    # Grow chords
    completed_chords = []
    failed_growths = 0

    print("\nGrowing chords...")
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
            min_length,
            max_length
        )

        if chord:
            completed_chords.append(chord)
        else:
            failed_growths += 1

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} starting points, "
                  f"completed {len(completed_chords)} chords")

    # Print final statistics
    stats = get_growth_stats(completed_chords)
    print(f"\nFinal Results:")
    print(f"Total chords: {len(completed_chords)}")
    print(f"Failed growths: {failed_growths}")
    print(f"Average length: {stats.avg_length:.1f}")
    print(f"Length range: {stats.min_length} to {stats.max_length}")
    print(f"Total points used: {stats.total_points_used}")

    return completed_chords


def get_growth_stats(chords: List[Set]) -> GrowthStats:
    """Calculate growth statistics."""
    if not chords:
        return GrowthStats(0, 0.0, 0, 0, 0)

    lengths = [len(c) for c in chords]
    total_points = sum(lengths)

    return GrowthStats(
        active_chords=len(chords),
        avg_length=np.mean(lengths),
        min_length=min(lengths),
        max_length=max(lengths),
        total_points_used=total_points
    )