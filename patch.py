from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from typing import List, Set, Dict, Tuple
import numpy.typing as npt


@dataclass
class PatchStats:
    total_patches: int
    avg_points_per_patch: float
    planarity_scores: List[float]


def vectorized_find_crossings(chords: List[Set],
                              positions: np.ndarray,
                              id_to_index: Dict[int, int],
                              max_distance: float = 3.0) -> List[Tuple[int, int, float]]:
    # Pre-compute chord data
    chord_data = []
    for chord in chords:
        points = np.array([positions[id_to_index[id(p)]] for p in chord])
        centroid = np.mean(points, axis=0)
        direction = points[-1] - points[0]
        direction /= np.linalg.norm(direction)
        chord_data.append((centroid, direction))

    centroids = np.array([c for c, _ in chord_data])
    directions = np.array([d for _, d in chord_data])

    # Find close pairs
    tree = cKDTree(centroids)
    pairs = tree.query_pairs(max_distance, output_type='ndarray')

    if len(pairs) == 0:
        return []

    # Compute angles between directions
    dir_products = np.abs(directions[pairs[:, 0]] * directions[pairs[:, 1]]).sum(axis=1)
    valid_angles = dir_products < 0.8

    valid_pairs = pairs[valid_angles]
    valid_dists = np.linalg.norm(centroids[valid_pairs[:, 0]] - centroids[valid_pairs[:, 1]], axis=1)

    return [(int(i), int(j), float(d)) for i, j, d in zip(valid_pairs[:, 0], valid_pairs[:, 1], valid_dists)]


def grow_patches_parallel(seed_crossings: List[Tuple[int, int, float]],
                          chords: List[Set],
                          positions: np.ndarray,
                          id_to_index: Dict[int, int],
                          target_size: int = 16) -> List[List[Set]]:
    # Pre-compute chord data
    chord_data = []
    for chord in chords:
        points = np.array([positions[id_to_index[id(p)]] for p in chord])
        chord_data.append({
            'points': points,
            'center': np.mean(points, axis=0),
            'dir': (points[-1] - points[0]) / np.linalg.norm(points[-1] - points[0]),
            'size': len(points)
        })

    centers = np.array([d['center'] for d in chord_data])
    dirs = np.array([d['dir'] for d in chord_data])
    sizes = np.array([d['size'] for d in chord_data])

    # Initialize all patches
    active_patches = []
    for seed_idx, cross_idx, _ in seed_crossings:
        normal = np.cross(dirs[seed_idx], dirs[cross_idx])
        if np.linalg.norm(normal) < 0.05:
            continue

        patch_size = sizes[seed_idx] + sizes[cross_idx]
        if patch_size > target_size * 1.5:
            continue

        active_patches.append({
            'chords': {seed_idx, cross_idx},
            'size': patch_size,
            'normal': normal / np.linalg.norm(normal),
            'center': (centers[seed_idx] * sizes[seed_idx] +
                       centers[cross_idx] * sizes[cross_idx]) / patch_size
        })

    # Grow all patches simultaneously until no more growth possible
    available = np.ones(len(chords), dtype=bool)
    for p in active_patches:
        for idx in p['chords']:
            available[idx] = False

    while True:
        added = False

        # Score all remaining chords for all patches at once
        for patch in active_patches:
            if patch['size'] >= target_size:
                continue

            dists = np.linalg.norm(centers - patch['center'], axis=1)
            plane_scores = np.abs(np.dot(centers - patch['center'], patch['normal']))
            dir_scores = np.abs(np.dot(dirs, patch['normal']))

            scores = dists + 2.0 * plane_scores + 5.0 * dir_scores
            size_mask = (patch['size'] + sizes) <= target_size * 1.2
            valid_mask = available & size_mask
            candidate_scores = np.where(valid_mask, scores, np.inf)

            best_idx = np.argmin(candidate_scores)
            if candidate_scores[best_idx] < np.inf:
                added = True
                available[best_idx] = False
                patch['chords'].add(best_idx)
                patch['size'] += sizes[best_idx]

                # Update patch center
                weight = sizes[best_idx] / patch['size']
                patch['center'] = (1 - weight) * patch['center'] + weight * centers[best_idx]

        if not added:
            break

    # Convert to final format
    return [[chords[i] for i in p['chords']]
            for p in active_patches
            if len(p['chords']) >= 3]


def generate_patches(z_chords: List[Set],
                     y_chords: List[Set],
                     x_chords: List[Set],
                     positions: np.ndarray,
                     id_to_index: Dict[int, int]) -> List[List[Set]]:
    all_chords = z_chords + y_chords + x_chords
    crossings = vectorized_find_crossings(all_chords, positions, id_to_index)
    return grow_patches_parallel(crossings, all_chords, positions, id_to_index)


from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np
import logging


@dataclass
class PatchMetrics:
    size: int
    planarity: float  # Average distance to patch plane
    compactness: float  # Average distance to center
    dir_alignment: float  # Average chord direction alignment
    growth_history: List[Tuple[int, float]]  # (chord_idx, score)


@dataclass
class GrowthStats:
    total_patches_started: int
    patches_completed: int
    avg_growth_iterations: float
    failed_additions: int
    size_distribution: Dict[int, int]
    planarity_scores: List[float]
    rejected_reasons: Dict[str, int]


def grow_patches_iterative(seed_crossings: List[Tuple[int, int, float]],
                           chords: List[Set],
                           positions: np.ndarray,
                           id_to_index: Dict[int, int],
                           target_size: int = 16,
                           debug: bool = True) -> Tuple[List[List[Set]], GrowthStats]:
    logging.info(f"Starting patch growth with {len(seed_crossings)} seed pairs")

    stats = GrowthStats(
        total_patches_started=0,
        patches_completed=0,
        avg_growth_iterations=0,
        failed_additions=0,
        size_distribution={},
        planarity_scores=[],
        rejected_reasons={'angle': 0, 'size': 0, 'no_candidates': 0}
    )

    # Initialize chord data with debug info
    chord_data = []
    for i, chord in enumerate(chords):
        points = np.array([positions[id_to_index[id(p)]] for p in chord])
        chord_data.append({
            'points': points,
            'center': np.mean(points, axis=0),
            'dir': (points[-1] - points[0]) / np.linalg.norm(points[-1] - points[0]),
            'size': len(points),
            'usage_count': 0,
            'rejected_count': 0
        })

    centers = np.array([d['center'] for d in chord_data])
    dirs = np.array([d['dir'] for d in chord_data])
    sizes = np.array([d['size'] for d in chord_data])

    # Initialize patches with metrics
    patches = []
    for seed_idx, cross_idx, dist in seed_crossings:
        normal = np.cross(dirs[seed_idx], dirs[cross_idx])
        normal_mag = np.linalg.norm(normal)

        if normal_mag < 0.05:
            stats.rejected_reasons['angle'] += 1
            continue

        stats.total_patches_started += 1
        patches.append({
            'chords': {seed_idx, cross_idx},
            'size': sizes[seed_idx] + sizes[cross_idx],
            'normal': normal / normal_mag,
            'center': (centers[seed_idx] * sizes[seed_idx] +
                       centers[cross_idx] * sizes[cross_idx]) /
                      (sizes[seed_idx] + sizes[cross_idx]),
            'metrics': PatchMetrics(
                size=sizes[seed_idx] + sizes[cross_idx],
                planarity=0.0,
                compactness=0.0,
                dir_alignment=abs(np.dot(dirs[seed_idx], dirs[cross_idx])),
                growth_history=[(seed_idx, 0.0), (cross_idx, 0.0)]
            )
        })

    available = np.ones(len(chords), dtype=bool)
    for p in patches:
        for idx in p['chords']:
            available[idx] = False
            chord_data[idx]['usage_count'] += 1

    # Growth iterations with detailed tracking
    max_iterations = target_size * 2
    total_iterations = 0

    for iteration in range(max_iterations):
        additions = 0

        for patch in patches:
            if patch['size'] >= target_size:
                continue

            dists = np.linalg.norm(centers - patch['center'], axis=1)
            plane_scores = np.abs(np.dot(centers - patch['center'], patch['normal']))
            dir_align = np.abs(np.dot(dirs, patch['normal']))

            scores = (dists / np.max(dists + 1e-6) +
                      2.0 * plane_scores / np.max(plane_scores + 1e-6) +
                      3.0 * dir_align)

            valid_mask = available & (sizes <= (target_size - patch['size']))
            scores = np.where(valid_mask, scores, np.inf)

            best_idx = np.argmin(scores)
            if scores[best_idx] == np.inf:
                stats.rejected_reasons['no_candidates'] += 1
                continue

            additions += 1
            available[best_idx] = False
            patch['chords'].add(best_idx)
            patch['size'] += sizes[best_idx]

            weight = sizes[best_idx] / patch['size']
            patch['center'] = ((1 - weight) * patch['center'] +
                               weight * centers[best_idx])

            # Update metrics
            patch['metrics'].growth_history.append((best_idx, scores[best_idx]))
            patch['metrics'].planarity = np.mean(plane_scores[list(patch['chords'])])
            patch['metrics'].compactness = np.mean(dists[list(patch['chords'])])
            patch['metrics'].dir_alignment = np.mean([abs(np.dot(dirs[i], patch['normal']))
                                                      for i in patch['chords']])

            chord_data[best_idx]['usage_count'] += 1

        if additions == 0:
            break

        total_iterations += 1

    # Collect final statistics
    completed_patches = [p for p in patches if len(p['chords']) >= 3]
    stats.patches_completed = len(completed_patches)
    stats.avg_growth_iterations = total_iterations / len(patches) if patches else 0

    for p in completed_patches:
        size = len(p['chords'])
        stats.size_distribution[size] = stats.size_distribution.get(size, 0) + 1
        stats.planarity_scores.append(p['metrics'].planarity)

    if debug:
        logging.info(f"Completed patch growth:")
        logging.info(f"Started: {stats.total_patches_started}, Completed: {stats.patches_completed}")
        logging.info(f"Size distribution: {stats.size_distribution}")
        logging.info(f"Average planarity: {np.mean(stats.planarity_scores):.3f}")
        logging.info(f"Rejected reasons: {stats.rejected_reasons}")

    return [[chords[i] for i in p['chords']] for p in completed_patches], stats