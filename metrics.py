import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from dataclasses import dataclass

import snic
@dataclass
class DriftMetrics:
    """Stores drift analysis results for a superpixel"""
    seed_pos: np.ndarray  # Initial position [z,y,x]
    final_pos: np.ndarray  # Final centroid position
    drift_vector: np.ndarray  # Vector from seed to final position
    drift_magnitude: float  # Euclidean distance moved
    intensity_delta: float  # Change in intensity from seed to final


def analyze_superpixel_drift(
        volume: np.ndarray,
        labels: np.ndarray,
        superpixels: List[snic.Superpixel],
        d_seed: int
) -> List[DriftMetrics]:
    """
    Analyze how far each superpixel centroid moved from its initial seed position.

    Parameters
    ----------
    volume : np.ndarray
        Original input volume (uint8)
    labels : np.ndarray
        Superpixel label array from SNIC
    superpixels : List[Superpixel]
        List of final superpixels
    d_seed : int
        Original seed spacing used

    Returns
    -------
    List[DriftMetrics]
        Drift analysis for each superpixel
    """
    lz, ly, lx = volume.shape
    drift_metrics = []

    # Calculate initial seed positions
    seed_positions = []
    for iz in range(0, lz, d_seed):
        for iy in range(0, ly, d_seed):
            for ix in range(0, lx, d_seed):
                seed_positions.append(np.array([iz, iy, ix]))

    # Match seeds to final superpixels
    for i, sp in enumerate(superpixels):
        if i >= len(seed_positions):
            break

        seed_pos = seed_positions[i]
        final_pos = np.array([sp.z, sp.y, sp.x])

        # Calculate drift vector and magnitude
        drift_vector = final_pos - seed_pos
        drift_magnitude = np.linalg.norm(drift_vector)

        # Calculate intensity change
        seed_intensity = float(volume[seed_pos[0], seed_pos[1], seed_pos[2]])
        intensity_delta = sp.c - seed_intensity

        drift_metrics.append(DriftMetrics(
            seed_pos=seed_pos,
            final_pos=final_pos,
            drift_vector=drift_vector,
            drift_magnitude=drift_magnitude,
            intensity_delta=intensity_delta
        ))

    return drift_metrics


def plot_drift_analysis(
        drift_metrics: List[DriftMetrics],
        volume_shape: Tuple[int, int, int],
        plot_type: str = 'magnitude'
) -> None:
    """
    Visualize superpixel drift analysis.

    Parameters
    ----------
    drift_metrics : List[DriftMetrics]
        Drift analysis for each superpixel
    volume_shape : Tuple[int, int, int]
        Shape of original volume (z,y,x)
    plot_type : str
        Type of plot to generate:
        - 'magnitude': Histogram of drift magnitudes
        - 'vectors': 3D scatter plot with drift vectors
        - 'heatmap': 2D heatmap of drift magnitudes (averaged over z)
    """
    if plot_type == 'magnitude':
        magnitudes = [m.drift_magnitude for m in drift_metrics]
        plt.figure(figsize=(10, 6))
        plt.hist(magnitudes, bins=30, edgecolor='black')
        plt.xlabel('Drift Magnitude (voxels)')
        plt.ylabel('Number of Superpixels')
        plt.title('Distribution of Superpixel Centroid Drift')

    elif plot_type == 'vectors':
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Plot drift vectors
        for metric in drift_metrics:
            ax.quiver(
                metric.seed_pos[2], metric.seed_pos[1], metric.seed_pos[0],  # start
                metric.drift_vector[2], metric.drift_vector[1], metric.drift_vector[0],  # direction
                color='b', alpha=0.6,
                length=1.0  # Scale vectors to actual length
            )

        ax.set_xlim(0, volume_shape[2])
        ax.set_ylim(0, volume_shape[1])
        ax.set_zlim(0, volume_shape[0])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Superpixel Drift Vectors')

    elif plot_type == 'heatmap':
        # Create 2D grid of drift magnitudes
        grid = np.zeros((volume_shape[1], volume_shape[2]))
        counts = np.zeros((volume_shape[1], volume_shape[2]))

        for metric in drift_metrics:
            y, x = int(metric.seed_pos[1]), int(metric.seed_pos[2])
            grid[y, x] = metric.drift_magnitude
            counts[y, x] += 1

        # Average where multiple superpixels overlap
        mask = counts > 0
        grid[mask] /= counts[mask]

        plt.figure(figsize=(10, 8))
        plt.imshow(grid, cmap='viridis')
        plt.colorbar(label='Average Drift Magnitude')
        plt.title('Spatial Distribution of Centroid Drift')
        plt.xlabel('X')
        plt.ylabel('Y')

    plt.show()


# Example usage:
def analyze_superpixel_movement(volume: np.ndarray, d_seed: int = 8, compactness: float = 40.0):
    """
    Run SNIC and analyze superpixel movement from initial seeds.

    Parameters
    ----------
    volume : np.ndarray
        Input volume (uint8)
    d_seed : int
        Seed spacing
    compactness : float
        SNIC compactness parameter
    """
    # Run SNIC
    labels, superpixels, _ = snic.run_snic(volume, d_seed, compactness)

    # Analyze drift
    drift_metrics = analyze_superpixel_drift(volume, labels, superpixels, d_seed)

    # Print summary statistics
    magnitudes = [m.drift_magnitude for m in drift_metrics]
    intensity_deltas = [abs(m.intensity_delta) for m in drift_metrics]

    print(f"Drift Statistics:")
    print(f"Average drift: {np.mean(magnitudes):.2f} voxels")
    print(f"Max drift: {np.max(magnitudes):.2f} voxels")
    print(f"Average intensity change: {np.mean(intensity_deltas):.2f}")

    # Generate plots
    for plot_type in ['magnitude', 'vectors', 'heatmap']:
        plot_drift_analysis(drift_metrics, volume.shape, plot_type)