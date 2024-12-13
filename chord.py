import numpy as np
from scipy.spatial import cKDTree
from collections import deque
from numba import jit
import numpy.typing as npt
from typing import Set, Dict, List, Tuple, Optional

@jit(nopython=True)
def _calculate_flow_direction(current_pos: npt.NDArray[np.float32],
                              current_c: float,
                              neighbor_pos: npt.NDArray[np.float32],
                              neighbor_c: float,
                              nearby_directions: npt.NDArray[np.float32],
                              gravity: float = 0.5,
                              gradient_weight: float = 1.0) -> npt.NDArray[np.float32]:
    """JIT-compiled flow calculation for better performance."""
    dp = neighbor_pos - current_pos
    gradient_acc = dp * (neighbor_c - current_c)
    gravity_acc = np.array([gravity, 0.0, 0.0])
    return gradient_acc * gradient_weight + gravity_acc + nearby_directions * 0.5

class ChordVector:
    """Memory-efficient physics properties of chord movement."""
    __slots__ = ('velocity', 'inertia', 'points')

    def __init__(self, dx: float = 0, dy: float = 0, dz: float = 0,
                 inertia: float = 0.8, max_history: int = 5):
        self.velocity = np.array([dx, dy, dz], dtype=np.float32)
        self.inertia = inertia
        self.points = deque(maxlen=max_history)

    def update(self, acceleration: npt.NDArray[np.float32], dt: float = 1.0) -> None:
        """Vectorized update with pre-allocated arrays."""
        self.velocity = np.multiply(np.add(self.velocity, np.multiply(acceleration, dt)),
                                    self.inertia, out=self.velocity)

    def predict_position(self, current_pos: npt.NDArray[np.float32],
                         dt: float = 1.0) -> npt.NDArray[np.float32]:
        """Optimized position prediction."""
        return np.add(current_pos, np.multiply(self.velocity, dt))

    def add_point(self, point: npt.NDArray[np.float32]) -> None:
        self.points.append(point)

    def get_direction(self) -> npt.NDArray[np.float32]:
        """Cached direction calculation."""
        if len(self.points) < 2:
            return self.velocity
        return np.subtract(self.points[-1], self.points[0])

def create_spatially_distributed_chords(superclusters: List,
                                        max_chords: int = 256,
                                        min_size: int = 128,
                                        max_size: int = 256,
                                        min_spacing: float = 10.0) -> Tuple[List[Set], Set]:
    """Optimized chord distribution with spatial indexing and vectorized operations."""

    # Pre-allocate arrays for better memory usage
    n_clusters = len(superclusters)
    cluster_positions = np.empty((n_clusters, 3), dtype=np.float32)

    # Vectorized position extraction
    for i, cluster in enumerate(superclusters):
        cluster_positions[i] = [cluster.z, cluster.y, cluster.x]

    # Use arrays instead of dictionaries for faster lookups
    id_to_index = {id(cluster): i for i, cluster in enumerate(superclusters)}
    available_mask = np.ones(n_clusters, dtype=bool)

    # Create spatial index once
    spatial_index = cKDTree(cluster_positions)

    # Vectorized grid creation
    bounds = np.array([
        [cluster_positions[:,0].min(), cluster_positions[:,0].max()],
        [cluster_positions[:,1].min(), cluster_positions[:,1].max()],
        [cluster_positions[:,2].min(), cluster_positions[:,2].max()]
    ])

    grid_size = int(np.sqrt(max_chords))
    grid_points = np.vstack(np.meshgrid(
        *[np.linspace(bounds[i,0], bounds[i,1], grid_size) for i in range(3)]
    )).reshape(3, -1).T

    # Batch process nearest neighbors
    distances, indices = spatial_index.query(grid_points, k=1)
    valid_indices = indices[distances < min_spacing * 2]
    np.random.shuffle(valid_indices)

    chords: List[Set] = []
    chord_trees: List[cKDTree] = []
    chord_directions: List[npt.NDArray[np.float32]] = []
    used_ids: Set[int] = set()

    # Pre-allocate arrays for distance calculations
    distance_buffer = np.empty(max_chords, dtype=np.float32)

    for start_idx in valid_indices:
        if len(chords) >= max_chords:
            break

        if not available_mask[start_idx]:
            continue

        current_chord: Set = set()
        current_cluster = superclusters[start_idx]
        current_idx = start_idx

        chord_physics = ChordVector()
        current_pos = cluster_positions[current_idx]

        while len(current_chord) < max_size:
            if not available_mask[current_idx]:
                break

            # Vectorized distance checking
            if chord_trees:
                for i, tree in enumerate(chord_trees):
                    distance_buffer[i] = tree.query(current_pos)[0]
                if np.any(distance_buffer[:len(chord_trees)] < min_spacing):
                    break

            current_chord.add(current_cluster)
            chord_physics.add_point(current_pos)
            available_mask[current_idx] = False
            used_ids.add(id(current_cluster))

            # Vectorized nearby direction calculation
            if chord_directions:
                directions_array = np.array(chord_directions)
                distances = np.linalg.norm(directions_array - current_pos, axis=1)
                nearby_mask = distances < min_spacing * 2
                avg_direction = (np.mean(directions_array[nearby_mask], axis=0)
                                 if nearby_mask.any() else np.zeros(3))
            else:
                avg_direction = np.zeros(3)

            # Optimize growth candidate selection using numpy operations
            best_candidate = None
            best_score = float('-inf')

            for neighbor, strength in current_cluster.connections.items():
                neighbor_idx = id_to_index.get(id(neighbor))
                if neighbor_idx is None or not available_mask[neighbor_idx]:
                    continue

                flow_acc = _calculate_flow_direction(
                    current_pos,
                    current_cluster.c,
                    np.array([neighbor.x, neighbor.y, neighbor.z]),
                    neighbor.c,
                    avg_direction
                )

                chord_physics.update(flow_acc)
                predicted_pos = chord_physics.predict_position(current_pos)

                neighbor_pos = np.array([neighbor.x, neighbor.y, neighbor.z])
                deviation = np.linalg.norm(neighbor_pos - predicted_pos)
                score = strength / (1 + deviation)

                if score > best_score:
                    best_score = score
                    best_candidate = (neighbor_idx, neighbor)

            if best_candidate is None:
                break

            current_idx, current_cluster = best_candidate
            current_pos = cluster_positions[current_idx]

        if len(current_chord) >= min_size:
            chords.append(current_chord)

            # Update spatial tracking structures
            chord_points = np.array([[p.z, p.y, p.x] for p in current_chord])
            chord_trees.append(cKDTree(chord_points))
            chord_directions.append(chord_physics.get_direction())

            available_count = np.sum(available_mask)
            print(f"Created chord {len(chords)} of size {len(current_chord)}")
            print(f"Available clusters: {available_count}")
            print(f"Used clusters: {len(used_ids)}")

    return chords, used_ids