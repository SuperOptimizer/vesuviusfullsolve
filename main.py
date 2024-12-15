from os import supports_effective_ids
import random
import skimage.exposure
import vtk
import matplotlib.cm as cm
import zarr
import numpy as np
import zarr
import time
import pipeline
import supervoxeler
import snic
from pathlib import Path

import metrics
import chord
import render
import patch

def preprocess(chunk, ISO, sharpen, min_component_size):

    print("Processing chunk...")
    processed = np.ascontiguousarray(chunk)
    mask = processed < ISO
    processed = skimage.filters.gaussian(processed, sigma=0.25)
    blurred = skimage.filters.gaussian(processed, sigma=0.5)
    unsharp_mask = processed - blurred
    processed = processed + sharpen * unsharp_mask
    processed = (processed - np.min(processed)) / (np.max(processed) - np.min(processed))
    processed = skimage.exposure.equalize_adapthist(processed,nbins=256,kernel_size=256)
    processed = (processed*255).astype(np.uint8)
    processed[mask] = 0
    processed = pipeline.apply_chunked_glcae_3d(processed)
    processed = pipeline.segment_and_clean(processed,ISO,ISO+32)

    #eroded = skimage.morphology.binary_erosion(processed > 0, footprint=skimage.morphology.ball(1))
    #dilated = skimage.morphology.binary_dilation(eroded > 0, footprint=skimage.morphology.ball(2))
    #processed[~dilated] = 0

    #eroded = skimage.morphology.binary_erosion(processed > 0, footprint=skimage.morphology.ball(2))
    #dilated = skimage.morphology.binary_dilation(eroded > 0, footprint=skimage.morphology.ball(1))
    #processed[~dilated] = 0

    binary = processed > 0
    labeled = skimage.measure.label(binary, connectivity=3)
    cleaned = skimage.morphology.remove_small_objects(labeled, min_size=min_component_size)
    processed[cleaned == 0] = 0

    return processed


def process_chunk(chunk_path, chunk_coords, chunk_dims, ISO, sharpen, min_component_size):
    print("Reading chunk...")
    scroll = zarr.open(chunk_path, mode="r")

    z_start, y_start, x_start = chunk_coords
    chunk = scroll[
            z_start:z_start + chunk_dims[0],
            y_start:y_start + chunk_dims[1],
            x_start:x_start + chunk_dims[2]
            ]
    processed = preprocess(chunk, ISO, sharpen, min_component_size)

    if np.count_nonzero(processed) == 0:
        raise ValueError("Preprocessing resulted in empty volume")

    print("Superclustering ...")
    start_time = time.time()
    labels, superclusters = snic.run_snic(processed, 2, 2 * 2 * 2, min_superpixel_size=2)

    if len(superclusters) == 0:
        raise ValueError("No superclusters generated")

    labels[processed == 0] = 0
    print(f"Superclustering completed in {time.time() - start_time:.2f} seconds")
    print(f"got {len(superclusters)} superclusters")
    print("\nDEBUG SNIC output:")
    print("First few superclusters:")
    for sc in superclusters[:5]:
        print(f"pos=({sc.z}, {sc.y}, {sc.x}), c={sc.c}")

    bounding_box = np.array([
        [0, chunk_dims[0]],
        [0, chunk_dims[1]],
        [0, chunk_dims[2]],
    ])

    zchords, ychords, xchords = chord.grow_fiber_chords(
        points=superclusters,
        bounds=bounding_box,
        min_length=4,
        max_length=16,
        num_chords=16384
    )

    if not (zchords or ychords or xchords):
        raise ValueError("No chords generated in any direction")

    print(f"got {len(zchords)} zchords")
    print(f"got {len(ychords)} ychords")
    print(f"got {len(xchords)} xchords")
    print(f"got {len(zchords + ychords + xchords)} total chords")

    positions, id_to_index, intensities = chord.initialize_points(superclusters, bounding_box)

    crossings = patch.vectorized_find_crossings(
        zchords + ychords + xchords,
        positions,
        id_to_index,
        max_distance=16.0
    )
    print(f"\nFound {len(crossings)} chord crossings")

    patches, stats = patch.grow_patches_iterative(
        seed_crossings=crossings,
        chords=zchords + ychords + xchords,
        positions=positions,
        id_to_index=id_to_index,
        target_size=12
    )

    print(f"\nGenerated {len(patches)} patches")
    print("\nPatch growth statistics:")
    print(f"Started: {stats.total_patches_started}, Completed: {stats.patches_completed}")
    print(f"Size distribution: {stats.size_distribution}")
    print(f"Average planarity: {np.mean(stats.planarity_scores):.3f}")
    print(f"Rejected reasons: {stats.rejected_reasons}")

    length_stats = chord.analyze_chord_lengths(patches, zchords, ychords, xchords, positions, id_to_index)
    print("\nChord length analysis:")
    chord.print_length_analysis(length_stats)

    if not patches:
        print("Warning: No patches generated, falling back to chord visualization")
        return visualize_chords(zchords, ychords, xchords, positions, id_to_index)

    return process_patches(patches, positions, id_to_index)


def visualize_chords(zchords, ychords, xchords, positions, id_to_index):
    """Fallback visualization using chords when no patches are generated"""
    all_centroids = []
    all_values = []
    all_indices = []

    for i, chord_set in enumerate([zchords, ychords, xchords]):
        for chord in chord_set:
            points = [(p.z, p.y, p.x) for p in chord]
            values = [p.c for p in chord]
            indices = [i * 100] * len(chord)  # Different color for each direction

            all_centroids.extend(points)
            all_values.extend(values)
            all_indices.extend(indices)

    if not all_centroids:
        raise ValueError("No visualization data available")

    return all_centroids, all_values, all_indices


def process_patches(patches, positions, id_to_index):
    """Process patches for visualization"""
    all_centroids = []
    all_values = []
    all_patch_indices = []

    for patch_idx, patch in enumerate(patches):
        for chord in patch:
            chord_points = [(p.z, p.y, p.x) for p in chord]
            chord_values = [p.c for p in chord]
            patch_indices = [patch_idx % 1024] * len(chord)

            all_centroids.extend(chord_points)
            all_values.extend(chord_values)
            all_patch_indices.extend(patch_indices)

    return all_centroids, all_values, all_patch_indices


def main():

    # Set up paths
    # keep if commented out
    SCROLL_PATH = Path("/Users/forrest/dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/1")
    #SCROLL_PATH = Path("/Users/forrest/dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/")
    #SCROLL_PATH = Path("/Volumes/vesuvius/dl.ash2txt.org/data/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/0")
    #SCROLL_PATH = Path("/Volumes/vesuvius/dl.ash2txt.org/data/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0")

    OUTPUT_DIR = Path("output")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Compiling Code...")
    supervoxeler.compile_supervoxeler()
    snic.compile_snic('./c/snic.c','./libsnic.so')


    chunk_coords=(2048, 2048, 2048)
    chunk_dims=(256,256,256)
    ISO=32
    sharpen=1
    min_component_size=128
    centroids, values, all_chord_indices = process_chunk(SCROLL_PATH, chunk_coords, chunk_dims, ISO, sharpen, min_component_size)


    # Prepare colors
    # Initialize all points as dark gray (51, 51, 51)
    r = np.full(len(centroids), 21, dtype=np.uint8)
    g = np.full(len(centroids), 21, dtype=np.uint8)
    b = np.full(len(centroids), 21, dtype=np.uint8)

    # Color patches using viridis
    patch_mask = np.array(all_chord_indices) >= 0
    if np.any(patch_mask):
        unique_patches = np.unique(all_chord_indices)
        for idx in unique_patches:
            if idx >= 0:  # Skip negative indices if any
                # Convert colormap output to numpy array before scaling and converting to uint8
                if idx < 256:
                    colormap = cm.viridis
                    color = np.array(colormap(idx / 255.0)[:3]) * 255
                elif idx < 512:
                    colormap = cm.magma
                    color = np.array(colormap((idx-256) / 255.0)[:3]) * 255
                elif idx < 768:
                    colormap = cm.inferno
                    color = np.array(colormap((idx-512) / 255.0)[:3]) * 255
                elif idx < 1024:
                    colormap = cm.cividis
                    color = np.array(colormap((idx-768) / 255.0)[:3]) * 255
                else:
                    colormap = cm.plasma
                    color = np.array(colormap((idx - 1024) / 255.0)[:3]) * 255


                color_u8 = color.astype(np.uint8)
                mask = np.array(all_chord_indices) == idx
                r[mask] = color_u8[0]
                g[mask] = color_u8[1]
                b[mask] = color_u8[2]
            else:
                color_u8 = np.array([20,20,20],dtype=np.uint8)
                mask = np.array(all_chord_indices) == idx
                r[mask] = color_u8[0]
                g[mask] = color_u8[1]
                b[mask] = color_u8[2]
                print()

    # Visualize results
    print("Visualizing results...")
    render.visualize_volume(centroids, values, all_chord_indices, r, g, b, 2)


if __name__ == "__main__":
    main()