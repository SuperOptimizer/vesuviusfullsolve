from dataclasses import dataclass
import numpy as np
from typing import List, Set, Tuple, Dict
from scipy.spatial import cKDTree

from metrics import DriftMetrics

@dataclass
class Patch:
    chords: List[Set]
    direction: np.ndarray
    centroid: np.ndarray
    bounds: Tuple[np.ndarray, np.ndarray]


def find_horizontal_chords(chords: List[Set],
                           positions: np.ndarray,
                           id_to_index: Dict[int, int],
                           drift_metrics: List[DriftMetrics],
                           min_xy_ratio: float = 1.5,
                           max_z_angle: float = 45) -> List[Set]:
    print(f"\nAnalyzing {len(chords)} chords for horizontal sections...")

    endpoints = np.array([[positions[id_to_index[id(list(chord)[0])]],
                           positions[id_to_index[id(list(chord)[-1])]]]
                          for chord in chords])

    directions = endpoints[:, 1] - endpoints[:, 0]

    xy_movement = np.sqrt(directions[:, 1] ** 2 + directions[:, 2] ** 2)
    z_movement = np.abs(directions[:, 0])

    mask = z_movement > 0.1
    xy_z_ratio = np.zeros_like(z_movement)
    xy_z_ratio[mask] = xy_movement[mask] / z_movement[mask]

    norms = np.linalg.norm(directions, axis=1)
    z_angles = np.degrees(np.arccos(np.clip(np.abs(directions[:, 0]) / norms, -1.0, 1.0)))

    drift_lookup = {tuple(m.final_pos): m.drift_magnitude for m in drift_metrics}
    chord_drifts = np.array([
        np.mean([drift_lookup.get(tuple(positions[id_to_index[id(p)]]), 0)
                 for p in chord])
        for chord in chords
    ])

    # Statistics before filtering
    print(
        f"XY/Z ratio stats: min={np.min(xy_z_ratio):.2f}, max={np.max(xy_z_ratio):.2f}, mean={np.mean(xy_z_ratio):.2f}")
    print(f"Z-angle stats: min={np.min(z_angles):.2f}, max={np.max(z_angles):.2f}, mean={np.mean(z_angles):.2f}")
    print(
        f"Drift stats: min={np.min(chord_drifts):.2f}, max={np.max(chord_drifts):.2f}, mean={np.mean(chord_drifts):.2f}")

    ratio_mask = xy_z_ratio > min_xy_ratio
    angle_mask = z_angles > max_z_angle
    drift_mask = chord_drifts > 1.5

    print(f"\nFilter results:")
    print(f"Chords passing XY/Z ratio filter: {np.sum(ratio_mask)} ({np.mean(ratio_mask) * 100:.1f}%)")
    print(f"Chords passing angle filter: {np.sum(angle_mask)} ({np.mean(angle_mask) * 100:.1f}%)")
    print(f"Chords passing drift filter: {np.sum(drift_mask)} ({np.mean(drift_mask) * 100:.1f}%)")

    horizontal_mask = ratio_mask & angle_mask & drift_mask
    horizontal = [chord for i, chord in enumerate(chords) if horizontal_mask[i]]

    return horizontal


def grow_patches(horizontal_chords: List[Set],
                 positions: np.ndarray,
                 id_to_index: Dict[int, int],
                 max_distance: float = 7.0,
                 min_parallel_score: float = 0.7) -> List[Patch]:
    print(f"\nGrowing patches from {len(horizontal_chords)} horizontal chords...")

    if not horizontal_chords or len(horizontal_chords) < 2:
        return []

    # Pre-compute chord data
    chord_data = []
    chord_lengths = []
    for chord in horizontal_chords:
        points = [positions[id_to_index[id(p)]] for p in chord]
        points_array = np.array(points)
        start, end = points_array[0], points_array[-1]
        direction = end - start
        direction = direction / np.linalg.norm(direction)
        centroid = np.mean(points_array, axis=0)
        chord_data.append((centroid, direction, points_array))
        chord_lengths.append(len(points))

    print(f"Chord length stats: min={min(chord_lengths)}, max={max(chord_lengths)}, mean={np.mean(chord_lengths):.1f}")

    centroids = np.array([data[0] for data in chord_data])
    directions = np.array([data[1] for data in chord_data])

    # Spatial analysis
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(max_distance, output_type='ndarray')

    print(f"Found {len(pairs)} potential chord pairs within {max_distance} units")

    if len(pairs) == 0:
        return []

    alignments = np.abs(np.sum(directions[pairs[:, 0]] * directions[pairs[:, 1]], axis=1))
    print(
        f"Alignment scores: min={np.min(alignments):.3f}, max={np.max(alignments):.3f}, mean={np.mean(alignments):.3f}")

    valid_pairs = pairs[alignments > min_parallel_score]
    print(f"Found {len(valid_pairs)} valid parallel pairs ({len(valid_pairs) / len(pairs) * 100:.1f}% of total)")

    if len(valid_pairs) == 0:
        return []

    # Union-find grouping
    n_chords = len(horizontal_chords)
    parents = np.arange(n_chords)
    ranks = np.zeros(n_chords, dtype=int)

    def find(x):
        if parents[x] != x:
            parents[x] = find(parents[x])
        return parents[x]

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if ranks[px] < ranks[py]:
            parents[px] = py
        elif ranks[px] > ranks[py]:
            parents[py] = px
        else:
            parents[py] = px
            ranks[px] += 1

    for i, j in valid_pairs:
        union(i, j)

    # Collect and analyze groups
    groups = {}
    for i in range(n_chords):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(i)

    group_sizes = [len(g) for g in groups.values()]
    print(f"\nPatch group stats:")
    print(f"Total groups: {len(groups)}")
    print(f"Group sizes: min={min(group_sizes)}, max={max(group_sizes)}, mean={np.mean(group_sizes):.1f}")

    # Create final patches
    patches = []
    for indices in groups.values():
        if len(indices) < 2:
            continue

        group_chords = [horizontal_chords[i] for i in indices]
        group_points = np.vstack([chord_data[i][2] for i in indices])

        patch = Patch(
            chords=group_chords,
            direction=np.mean(directions[indices], axis=0),
            centroid=np.mean(group_points, axis=0),
            bounds=(np.min(group_points, axis=0), np.max(group_points, axis=0))
        )
        patches.append(patch)

    print(f"\nCreated {len(patches)} final patches")
    return patches


def analyze_patches(patches: List[Patch], volume_shape: Tuple[int, int, int]) -> None:
    if not patches:
        print("\nNo patches found")
        return

    total_chords = sum(len(p.chords) for p in patches)
    avg_chords = total_chords / len(patches)

    patch_sizes = np.array([p.bounds[1] - p.bounds[0] for p in patches])
    patch_volumes = np.prod(patch_sizes, axis=1)

    print(f"\nDetailed patch analysis:")
    print(f"Total patches: {len(patches)}")
    print(f"Total chords in patches: {total_chords}")
    print(f"Chords per patch: min={min(len(p.chords) for p in patches)}, "
          f"max={max(len(p.chords) for p in patches)}, mean={avg_chords:.1f}")
    print(f"Patch dimensions (z,y,x):")
    print(f"  Min: {np.min(patch_sizes, axis=0)}")
    print(f"  Max: {np.max(patch_sizes, axis=0)}")
    print(f"  Mean: {np.mean(patch_sizes, axis=0)}")
    print(f"Patch volumes: min={np.min(patch_volumes):.1f}, "
          f"max={np.max(patch_volumes):.1f}, mean={np.mean(patch_volumes):.1f}")