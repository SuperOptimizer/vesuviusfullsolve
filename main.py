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
import path
import volume

def preprocess(chunk, ISO, sharpen, min_component_size):

    print("Processing chunk...")
    processed = np.ascontiguousarray(chunk)
    mask = processed < ISO
    processed = skimage.filters.gaussian(processed, sigma=1).astype(np.float32)
    blurred = skimage.filters.gaussian(processed, sigma=2).astype(np.float32)
    unsharp_mask = processed - blurred
    processed = processed + sharpen * unsharp_mask
    processed = (processed - np.min(processed)) / (np.max(processed) - np.min(processed))
    processed = skimage.exposure.equalize_hist(processed,nbins=256).astype(np.float32)
    processed = (processed*255).astype(np.uint8)
    processed[mask] = 0
    processed = pipeline.apply_chunked_glcae_3d(processed)
    processed = pipeline.segment_and_clean(processed,ISO,ISO+32)

    eroded = skimage.morphology.binary_erosion(processed > 0, footprint=skimage.morphology.ball(1))
    dilated = skimage.morphology.binary_dilation(eroded > 0, footprint=skimage.morphology.ball(1))
    processed[~dilated] = 0

    #eroded = skimage.morphology.binary_erosion(processed > 0, footprint=skimage.morphology.ball(2))
    #dilated = skimage.morphology.binary_dilation(eroded > 0, footprint=skimage.morphology.ball(1))
    #processed[~dilated] = 0

    binary = processed > 0
    labeled = skimage.measure.label(binary, connectivity=3)
    cleaned = skimage.morphology.remove_small_objects(labeled, min_size=min_component_size)
    processed[cleaned == 0] = 0

    return processed


def process_chunk(cluster_vol, chunk_path, chunk_coords, chunk_dims, padding, ISO, sharpen, min_component_size, d_seed, compactness):
    print("Reading chunk...")
    scroll = zarr.open(chunk_path, mode="r")

    z_start, y_start, x_start = chunk_coords
    chunk = scroll[
            z_start-padding[0]:z_start + chunk_dims[0] + padding[0],
            y_start-padding[1]:y_start + chunk_dims[1] + padding[1],
            x_start-padding[2]:x_start + chunk_dims[2] + padding[2]
            ]
    print(f"getting chunk at z={z_start-padding[0]}:{z_start + chunk_dims[0] + padding[0]}")
    print(f"                 y={y_start-padding[1]}:{y_start + chunk_dims[1] + padding[1]}")
    print(f"                 x={x_start-padding[2]}:{x_start + chunk_dims[2] + padding[2]}")
    print(f"z len = {chunk_dims[0] + padding[0]*2} ")
    print(f"y len = {chunk_dims[1] + padding[1]*2} ")
    print(f"x len = {chunk_dims[2] + padding[2]*2} ")
    processed = preprocess(chunk, ISO, sharpen, min_component_size)
    #processed[processed > 0] = 255
    if np.count_nonzero(processed) == 0:
        raise ValueError("Preprocessing resulted in empty volume")

    print("Superclustering ...")
    start_time = time.time()
    labels, superclusters = snic.run_snic(processed, d_seed, compactness, min_superpixel_size=2)

    # Calculate chunk indices based on the chunk coordinates
    z_idx = z_start // 128
    y_idx = y_start // 128
    x_idx = x_start // 128

    # Write the superclusters
    cluster_vol.write_superclusters(superclusters, (z_idx, y_idx, x_idx))

    centroids = [(sp.z,sp.y,sp.x) for sp in superclusters]
    print(f"smallest distance between 2 superpixels: {metrics.find_min_centroid_distance(centroids)}")

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

    zpaths, ypaths, xpaths = path.grow_paths_parallel(
        points=superclusters,
        bounds=bounding_box,
        num_paths=4096,  # Reduced number for longer paths
        min_length=8,  # Increased minimum length
        max_length=256
    )

    return visualize_paths(zpaths,[],[])

    zchords, ychords, xchords = chord.grow_fiber_chords(
        points=superclusters,
        bounds=bounding_box,
        min_length=4,
        max_length=8,
        num_chords=32768
    )




    if not (zchords or ychords or xchords):
        raise ValueError("No chords generated in any direction")

    print(f"got {len(zchords)} zchords")
    print(f"got {len(ychords)} ychords")
    print(f"got {len(xchords)} xchords")
    print(f"got {len(zchords + ychords + xchords)} total chords")

    positions, id_to_index, intensities = chord.initialize_points(superclusters, bounding_box)
    chord.print_length_analysis(chord.analyze_chord_lengths(zchords,ychords,xchords,positions,id_to_index))

    return visualize_chords(zchords, ychords, xchords)



def visualize_chords(zchords, ychords, xchords):
    """Fallback visualization using chords when no patches are generated"""
    all_centroids = []
    all_values = []
    all_indices = []

    for i, chord_set in enumerate([zchords, ychords, xchords]):
        for j, chord_ in enumerate(chord_set):
            points = [(p.z, p.y, p.x) for p in chord_]
            values = [p.c for p in chord_]
            indices = [j%1024] * len(chord_)

            all_centroids.extend(points)
            all_values.extend(values)
            all_indices.extend(indices)

    if not all_centroids:
        raise ValueError("No visualization data available")

    return all_centroids, all_values, all_indices


def visualize_paths(zpaths, ypaths, xpaths):
    """Visualize the longer continuous paths with different colors for each direction"""
    all_centroids = []
    all_values = []
    all_indices = []

    # Assign different base indices for each direction to create distinct color ranges
    direction_offsets = {
        'z': 0,  # Will use viridis colormap
        'y': 256,  # Will use magma colormap
        'x': 512  # Will use inferno colormap
    }

    for direction, paths in [('z', zpaths), ('y', ypaths), ('x', xpaths)]:
        base_idx = direction_offsets[direction]

        for i, path in enumerate(paths):
            # Ensure we don't exceed color index limits
            path_idx = base_idx + (i % 256)

            # Extract points from path
            points = [(p.z, p.y, p.x) for p in path]
            values = [p.c for p in path]
            indices = [path_idx] * len(path)

            all_centroids.extend(points)
            all_values.extend(values)
            all_indices.extend(indices)

    if not all_centroids:
        raise ValueError("No visualization data available")

    return all_centroids, all_values, all_indices

def main():

    # Set up paths
    # keep if commented out
    #SCROLL_PATH = Path("/Users/forrest/dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/1")
    #SCROLL_PATH = Path("/Users/forrest/dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/")
    SCROLL_PATH = Path("/Volumes/vesuvius/dl.ash2txt.org/data/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/0")
    #SCROLL_PATH = Path("/Volumes/vesuvius/dl.ash2txt.org/data/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0")
    ZARR_PATH = "/Volumes/vesuvius/clusters.zarr"
    OUTPUT_DIR = Path("output")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Compiling Code...")
    supervoxeler.compile_supervoxeler()
    snic.compile_snic('./c/snic.c','./libsnic.so')

    #the cluster volume stores the superclusters as z y x point spheres with a color and size
    cluster_vol = volume.ClusterVolume(ZARR_PATH,'w',True)
    #we pad this many voxels on either side of the chunk in order to have good cluster coverage over the entire volume
    #so that we can follow chords and paths outside of one chunk into another one easier
    padding = (4,4,4)
    #the z y x of where to start in the volume
    chunk_coords=(2048, 3072, 4096)
    #the z y x dimensions of the chunk of data we're looking at
    chunk_dims=(128,128,128)
    #ISO for the standard zarr volume. multiple of 8 as standard zarr volumes zero out the low 3 bits
    ISO=8
    #unsharp masking multiplier
    sharpen=1
    #min amount of adjacent voxels to consider as matter and not noise
    min_component_size=8
    # how much we seed superpixels when, e.g. every 2 z y or x values. SHOULD BE 2!!! probably wont work with any other value
    d_seed = 2
    #how compact the superpixeling should be. should be around ~128 to give superpixel of max size=255
    compactness = 128
    centroids, values, all_chord_indices = process_chunk(cluster_vol, SCROLL_PATH, chunk_coords, chunk_dims, padding,ISO, sharpen, min_component_size, d_seed, compactness)


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