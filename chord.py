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
    chord: Set
    position: np.ndarray
    direction: np.ndarray
    intensity: float  # Store intensity for gradient computation


def initialize_coordinates(points: List, bounds: npt.NDArray[np.float32]) -> Tuple[
    np.ndarray, Dict[int, int], np.ndarray]:
    """Convert point positions to coordinate system and extract intensities."""
    n_points = len(points)
    positions = np.empty((n_points, 3), dtype=np.float32)
    intensities = np.empty(n_points, dtype=np.float32)

    print(f"\nInitializing {n_points} points")
    print(f"Bounds: {bounds}")

    for i, point in enumerate(points):
        positions[i] = [float(point.z), float(point.y), float(point.x)]
        intensities[i] = float(point.c)  # Assuming 'c' is the intensity value

    id_to_index = {id(point): i for i, point in enumerate(points)}
    return positions, id_to_index, intensities


def estimate_local_gradient(pos: np.ndarray,
                            spatial_index: cKDTree,
                            point_positions: np.ndarray,
                            intensities: np.ndarray,
                            radius: float = 3.0,
                            z_bias: float = 0.7) -> np.ndarray:
    """
    Estimate intensity gradient at a point with configurable z-direction bias.

    Args:
        pos: Position to estimate gradient at
        spatial_index: KD-tree for spatial lookups
        point_positions: Array of all point positions
        intensities: Array of intensity values
        radius: Search radius for gradient estimation
        z_bias: Weight for z-direction bias (0-1)

    Returns:
        Normalized gradient vector
    """
    nearby_indices = spatial_index.query_ball_point(pos, radius)
    if len(nearby_indices) < 4:  # Need minimum points for gradient
        return np.array([1.0, 0.0, 0.0])  # Default to z direction

    # Calculate weighted gradient
    gradient = np.zeros(3)
    center_intensity = intensities[spatial_index.query(pos)[1]]

    for idx in nearby_indices:
        dp = point_positions[idx] - pos
        dist = np.linalg.norm(dp)
        if dist < 0.1:
            continue

        # Weight by distance and intensity difference
        weight = 1.0 / (dist + 0.1)
        intensity_diff = intensities[idx] - center_intensity
        gradient += weight * intensity_diff * (dp / dist)

    # Normalize gradient
    grad_norm = np.linalg.norm(gradient)
    if grad_norm > 0:
        gradient = gradient / grad_norm
    else:
        gradient = np.array([1.0, 0.0, 0.0])

    # Apply configurable z-direction bias
    gradient = z_bias * np.array([1.0, 0.0, 0.0]) + (1.0 - z_bias) * gradient
    return gradient / np.linalg.norm(gradient)


def find_next_point(current_state: ChordState,
                    spatial_index: cKDTree,
                    point_positions: np.ndarray,
                    intensities: np.ndarray,
                    available_mask: np.ndarray,
                    nearby_chords: List[ChordState],
                    id_to_index: Dict[int, int],
                    search_radius: float = 5.0,
                    min_direction_score: float = 0.5,
                    spatial_weight: float = 0.4,
                    min_neighbor_dist: float = 3.0) -> Optional[Tuple[int, np.ndarray, float, float]]:
    """
    Find the next best point with enhanced spatial organization and neighbor avoidance.

    Args:
        current_state: Current state of the growing chord
        spatial_index: KD-tree for spatial lookups
        point_positions: Array of all point positions
        intensities: Array of intensity values
        available_mask: Boolean mask of available points
        nearby_chords: List of nearby growing chords
        id_to_index: Mapping from point ID to index
        search_radius: Radius for searching neighboring points
        min_direction_score: Minimum acceptable direction alignment score
        spatial_weight: Weight for spatial organization score (0-1)
        min_neighbor_dist: Minimum desired distance to neighboring chords

    Returns:
        Optional tuple of (index, position, score, intensity)
    """
    nearby_indices = spatial_index.query_ball_point(current_state.position, search_radius)

    if not nearby_indices:
        return None

    # Get local gradient with higher z-bias for better upward growth
    local_gradient = estimate_local_gradient(
        current_state.position,
        spatial_index,
        point_positions,
        intensities,
        z_bias=0.8
    )

    # Get first point's position using id_to_index
    first_point = list(current_state.chord)[0]
    first_point_idx = id_to_index[id(first_point)]
    chord_start_xy = point_positions[first_point_idx][1:]  # Get YX coordinates

    candidates = []
    for idx in nearby_indices:
        if not available_mask[idx]:
            continue

        pos = point_positions[idx]
        dp = pos - current_state.position
        distance = np.linalg.norm(dp)

        if distance < 0.1:  # Skip points too close to current position
            continue

        dp_normalized = dp / distance

        # Basic direction score considering gradient alignment
        direction_score = np.dot(dp_normalized, local_gradient)
        if direction_score < min_direction_score:
            continue

        # Calculate spatial organization score
        xy_pos = pos[1:]  # Current YX position

        # Spatial score based on maintaining original XY position
        spatial_deviation = np.linalg.norm(xy_pos - chord_start_xy)
        spatial_score = 1.0 / (1.0 + spatial_deviation * 0.1)

        # Enhanced neighbor avoidance
        repulsion_score = 1.0
        min_neighbor_distance = float('inf')

        for other_chord in nearby_chords:
            if other_chord.chord & current_state.chord:  # Skip our own chord
                continue

            # Calculate distance to neighbor's current position
            other_dist = np.linalg.norm(other_chord.position[1:] - xy_pos)
            min_neighbor_distance = min(min_neighbor_distance, other_dist)

            # Apply stronger repulsion for closer neighbors
            if other_dist < min_neighbor_dist:
                repulsion_factor = other_dist / min_neighbor_dist
                repulsion_score *= repulsion_factor * repulsion_factor  # Quadratic falloff

        # Consider intensity similarity for smoother paths
        intensity_diff = abs(intensities[idx] - current_state.intensity)
        intensity_score = 1.0 / (1.0 + intensity_diff * 0.1)

        # Z-progress score to encourage upward growth
        z_progress = max(0, (pos[0] - current_state.position[0]) / distance)
        z_score = 0.5 + 0.5 * z_progress  # Baseline of 0.5, bonus for upward movement

        # Combined score with weighted components
        total_score = (
                direction_score * 0.25 +  # Basic direction alignment
                spatial_score * 0.25 +  # Maintaining original XY position
                repulsion_score * 0.2 +  # Avoiding neighbors
                intensity_score * 0.15 +  # Intensity continuity
                z_score * 0.15  # Upward progress
        )

        # Apply additional penalty for very close neighbors
        if min_neighbor_distance < min_neighbor_dist * 0.5:
            total_score *= 0.5

        candidates.append((idx, pos, total_score, intensities[idx]))

    if not candidates:
        return None

    # Return the candidate with the highest score
    return max(candidates, key=lambda x: x[2])


def find_nearby_chords(current_pos: np.ndarray,
                       active_chords: List[ChordState],
                       max_distance: float = 10.0) -> List[ChordState]:
    """Find chords that are growing nearby."""
    nearby = []
    for chord in active_chords:
        if np.linalg.norm(chord.position - current_pos) < max_distance:
            nearby.append(chord)
    return nearby


def grow_fiber_chords(points: List,
                      bounds: npt.NDArray[np.float32],
                      num_chords: int = 256,
                      min_length: int = 8,
                      max_length: int = 32,
                      z_bias: float = 0.8,
                      min_direction_score: float = 0.4,
                      search_radius: float = 5.0) -> List[Set]:
    """Generate multiple fiber-like chords with enhanced spatial organization."""

    # Initialize coordinate system
    point_positions, id_to_index, intensities = initialize_coordinates(points, bounds)

    # Create spatial index and available mask
    available_mask = np.ones(len(points), dtype=bool)
    spatial_index = cKDTree(point_positions)

    print(f"\nFinding starting points for {num_chords} chords...")

    # Calculate XY grid size for even distribution
    xy_cells = int(np.sqrt(num_chords))
    x_step = (bounds[2, 1] - bounds[2, 0]) / xy_cells
    y_step = (bounds[1, 1] - bounds[1, 0]) / xy_cells

    # Find starting points close to z=0 distributed in XY plane
    starting_indices = []
    z_tolerance = 5.0  # Initial Z tolerance
    min_intensity = np.percentile(intensities, 30)  # Minimum intensity threshold

    while len(starting_indices) < num_chords and z_tolerance < bounds[0, 1]:
        for i in range(xy_cells):
            for j in range(xy_cells):
                if len(starting_indices) >= num_chords:
                    break

                # Calculate target XY position
                x = bounds[2, 0] + (i + 0.5 + np.random.uniform(-0.3, 0.3)) * x_step
                y = bounds[1, 0] + (j + 0.5 + np.random.uniform(-0.3, 0.3)) * y_step

                # Find closest point to (0,y,x) within z_tolerance
                nearby_points = spatial_index.query_ball_point(
                    [0, y, x],
                    r=z_tolerance
                )

                # Filter by availability and intensity
                candidates = [
                    idx for idx in nearby_points
                    if available_mask[idx]
                       and intensities[idx] > min_intensity
                ]

                if candidates:
                    # Select the point closest to z=0
                    z_coords = point_positions[candidates][:, 0]
                    best_idx = candidates[np.argmin(z_coords)]
                    starting_indices.append(best_idx)
                    available_mask[best_idx] = False

        # Increase tolerance if we need more points
        z_tolerance *= 1.5

    print(f"Found {len(starting_indices)} starting points")
    print(f"Z-range of starting points: {np.min(point_positions[starting_indices][:, 0]):.1f} "
          f"to {np.max(point_positions[starting_indices][:, 0]):.1f}")

    # Initialize chords
    active_chords = []
    for idx in starting_indices:
        initial_gradient = estimate_local_gradient(
            point_positions[idx],
            spatial_index,
            point_positions,
            intensities,
            z_bias=z_bias
        )

        chord = {points[idx]}
        active_chords.append(ChordState(
            chord=chord,
            position=point_positions[idx],
            direction=initial_gradient,
            intensity=intensities[idx]
        ))

    print(f"Successfully initialized {len(active_chords)} chords")

    # Main growth loop with enhanced spatial organization
    completed_chords = []
    iteration = 0

    while active_chords and iteration < max_length * 2:
        next_chords = []

        for state in active_chords:
            if len(state.chord) >= max_length:
                if len(state.chord) >= min_length:
                    completed_chords.append(state.chord)
                continue

            # Find nearby chords for parallel growth and spatial organization
            nearby_chords = find_nearby_chords(state.position, active_chords)

            result = find_next_point(
                state,
                spatial_index,
                point_positions,
                intensities,
                available_mask,
                nearby_chords,
                id_to_index,
                search_radius=search_radius,
                min_direction_score=min_direction_score
            )

            if result is None:
                if len(state.chord) >= min_length:
                    completed_chords.append(state.chord)
                continue

            idx, new_pos, score, intensity = result
            point = points[idx]

            # Add point to chord
            state.chord.add(point)
            available_mask[idx] = False

            # Update direction with spatial organization consideration
            new_dir = new_pos - state.position
            new_dir = new_dir / np.linalg.norm(new_dir)

            local_gradient = estimate_local_gradient(
                new_pos,
                spatial_index,
                point_positions,
                intensities,
                z_bias=z_bias
            )

            # Get initial XY position of this chord
            first_point_id = id(list(state.chord)[0])
            first_point_idx = id_to_index[first_point_id]
            current_xy = new_pos[1:]

            # Calculate direction to original XY position
            start_xy = point_positions[first_point_idx][1:]
            xy_correction = np.array([0, *(start_xy - current_xy)])
            if np.linalg.norm(xy_correction) > 0:
                xy_correction = xy_correction / np.linalg.norm(xy_correction)

            # Enhanced direction smoothing with spatial organization
            if nearby_chords:
                nearby_directions = np.array([c.direction for c in nearby_chords])
                mean_direction = np.mean(nearby_directions, axis=0)
                mean_direction = mean_direction / np.linalg.norm(mean_direction)

                smoothed_dir = (
                    0.3 * state.direction +      # Previous direction
                    0.2 * new_dir +              # New direction
                    0.2 * local_gradient +       # Local gradient
                    0.2 * xy_correction +        # Pull toward original XY
                    0.1 * mean_direction         # Influence from neighbors
                )
            else:
                smoothed_dir = (
                    0.3 * state.direction +
                    0.3 * new_dir +
                    0.2 * local_gradient +
                    0.2 * xy_correction
                )

            smoothed_dir = smoothed_dir / np.linalg.norm(smoothed_dir)

            next_chords.append(ChordState(
                chord=state.chord,
                position=new_pos,
                direction=smoothed_dir,
                intensity=intensity
            ))

        active_chords = next_chords

        if iteration % 5 == 0:
            all_chords = completed_chords + [c.chord for c in active_chords]
            stats = get_growth_stats(all_chords)
            print(f"\nIteration {iteration}:")
            print(f"Active chords: {len(active_chords)}")
            print(f"Completed chords: {len(completed_chords)}")
            print(f"Average length: {stats.avg_length:.1f}")
            print(f"Length range: {stats.min_length} to {stats.max_length}")

        iteration += 1

    # Add remaining active chords that meet minimum length
    final_chords = completed_chords + [c.chord for c in active_chords if len(c.chord) >= min_length]

    # Final statistics
    final_stats = get_growth_stats(final_chords)
    print(f"\nFinal Results:")
    print(f"Total chords: {len(final_chords)}")
    print(f"Average length: {final_stats.avg_length:.1f}")
    print(f"Length range: {final_stats.min_length} to {final_stats.max_length}")
    print(f"Total points used: {final_stats.total_points_used}")

    return final_chords


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