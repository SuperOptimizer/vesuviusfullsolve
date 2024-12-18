from os import supports_effective_ids
import random
import skimage.exposure
import vtk
import matplotlib.cm as cm
import zarr
import numpy as np
import zarr
import time
import multiprocessing
from conda.common.serialize import yaml_safe_dump
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import pipeline
import supervoxeler
import snic
from pathlib import Path

import metrics
import chord
import render
import path
import volume
import vis

#NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)
NUM_WORKERS = 1

from numcodecs import Blosc

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

def process_chunk_wrapper(args):
    """Wrapper function to unpack arguments for process_chunk"""
    (scroll_zarr, fiber_zarr, cluster_zarr, chord_zarr, segment_zarr,
     z, y, x, dims, padding, ISO, sharpen, min_component_size, d_seed, compactness) = args
    return process_chunk(
        scroll_zarr, fiber_zarr, cluster_zarr, chord_zarr, segment_zarr,
        z, y, x, dims, padding, ISO, sharpen, min_component_size, d_seed, compactness
    )

def process_chunk(scroll_zarr, fiber_zarr, cluster_zarr, chord_zarr, segment_zarr, z, y, x, dims, padding, ISO, sharpen, min_component_size, d_seed, compactness):
    slice_ = (
        slice(z - padding[0], z + dims[0] + padding[0]),
        slice(y - padding[1], y + dims[1] + padding[1]),
        slice(x - padding[2], x + dims[2] + padding[2])
    )
    fiber_chunk = fiber_zarr[slice_]
    scroll_chunk = scroll_zarr[slice_]
    if np.all(fiber_chunk == 0):
        print(f"skipping chunk at {z},{y},{x} because no fiber")
        return
    print(f"processing chunk at {z},{y},{x}")

    scroll_chunk_processed = preprocess(scroll_chunk, ISO, sharpen, min_component_size)
    labels, superclusters = snic.run_snic(scroll_chunk_processed, d_seed, compactness, min_superpixel_size=2)
    print(f"writing supercluster chunk list to {z//128},{y//128},{x//128}")
    cluster_zarr[z//128,y//128,x//128] = volume.superclusters_to_numpy(superclusters)

    bounding_box = np.array([
        [0, dims[0]],
        [0, dims[1]],
        [0, dims[2]],
    ])

    zpaths = path.grow_paths_parallel(
        points=superclusters,
        bounds=bounding_box,
        axis=0,
        num_paths=path.MAX_PATHS,  # Reduced number for longer paths
        min_length=path.MIN_PATH_LENGTH,  # Increased minimum length
        max_length=path.MAX_PATH_LENGTH
    )
    print(f"got {len(zpaths)} zpaths")
    if len(zpaths) > 0:
        path_data = volume.paths_to_numpy(zpaths, superclusters, path.MAX_PATHS, path.MAX_PATH_LENGTH)
        chord_zarr[z//128,y//128,x//128] = path_data

        #centroids, values, indices = vis.visualize_paths(zpaths, [], [])
        #r, g, b = colorize(centroids, indices)
        #render.visualize_volume(centroids, values, indices, r, g, b, 2)



def do_all_superpixeling(scroll_path, fiber_path, cluster_path, chord_path, segment_path):

    scroll_synchronizer = zarr.ProcessSynchronizer("/Volumes/vesuvius/scroll1a_volume.sync")
    fiber_synchronizer = zarr.ProcessSynchronizer("/Volumes/vesuvius/scroll1a_fiber.sync")
    cluster_synchronizer = zarr.ProcessSynchronizer("/Volumes/vesuvius/scroll1a_cluster.sync")
    chord_synchronizer = zarr.ProcessSynchronizer("/Volumes/vesuvius/scroll1a_chord.sync")
    segment_synchronizer = zarr.ProcessSynchronizer("/Volumes/vesuvius/scroll1a_segment.sync")

    compressor = Blosc(cname='blosclz', clevel=9)

    scroll_zarr = zarr.open(scroll_path, mode="r",compressor=compressor,synchronizer=scroll_synchronizer)
    fiber_zarr = zarr.open(fiber_path, mode="r",compressor=compressor,synchronizer=fiber_synchronizer)
    cluster_zarr = zarr.open(cluster_path,mode='w',
                             shape=(113, 63, 64, 64 * 64 * 64, 5),
                            chunks=(1, 1, 1, 64 * 64 * 64, 5),compressor=compressor,synchronizer=cluster_synchronizer)
    chord_zarr = zarr.open(chord_path,mode='w',
                             shape=(113, 63, 64, path.MAX_PATHS, path.MAX_PATH_LENGTH),
                            chunks=(1, 1, 1, path.MAX_PATHS, path.MAX_PATH_LENGTH),compressor=compressor,synchronizer=chord_synchronizer)
    segment_zarr = zarr.open(segment_path, mode="w", shape=scroll_zarr.shape,chunks=scroll_zarr.chunks,compressor=compressor,synchronizer=segment_synchronizer)

    #dimensions of the chunk
    dims = (128,128,128)

    # we pad this many voxels on either side of the chunk in order to have good cluster coverage over the entire volume
    # so that we can follow chords and paths outside of one chunk into another one easier
    padding = (4, 4, 4)

    # ISO for the standard zarr volume. multiple of 8 as standard zarr volumes zero out the low 3 bits
    ISO = 16

    # unsharp masking multiplier
    sharpen = 1

    # min amount of adjacent voxels to consider as matter and not noise
    min_component_size = 4

    # how much we seed superpixels when, e.g. every 2 z y or x values. SHOULD BE 2!!! probably wont work with any other value
    d_seed = 2

    # how compact the superpixeling should be. should be around ~256 to give superpixel of max size around 255
    compactness = 256


    # Generate all chunk coordinates
    chunk_coords = []
    for z in range(2048, scroll_zarr.shape[0]-1024, dims[0]):
        for y in range(2048, scroll_zarr.shape[1]-1024, dims[1]):
            for x in range(2048, scroll_zarr.shape[2]-1024, dims[2]):
                chunk_coords.append((z, y, x))

    # Prepare arguments for each chunk
    chunk_args = [
        (scroll_zarr, fiber_zarr, cluster_zarr, chord_zarr, segment_zarr,
         z, y, x, dims, padding, ISO, sharpen, min_component_size, d_seed, compactness)
        for z, y, x in chunk_coords
    ]

    # Use number of CPU cores minus 1 to avoid overloading
    print(f"Processing {len(chunk_coords)} chunks using {NUM_WORKERS} workers")

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks and create a map of futures to their chunk coordinates
        future_to_coords = {
            executor.submit(process_chunk_wrapper, args): (args[5], args[6], args[7])  # z,y,x coords
            for args in chunk_args
        }

        # Process completed futures as they finish
        for future in concurrent.futures.as_completed(future_to_coords):
            z, y, x = future_to_coords[future]
            try:
                future.result()  # Will raise exception if task failed
                print(f"Successfully processed chunk at {z},{y},{x}")
            except Exception as e:
                print(f"Chunk at {z},{y},{x} failed with error: {str(e)}")
                # Optionally log the full traceback
                import traceback
                print(f"Full traceback:\n{traceback.format_exc()}")


def colorize(centroids, all_chord_indices):

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
    return r,g,b

def main():

    # Set up paths
    # keep if commented out
    #SCROLL_PATH = Path("/Users/forrest/dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/1")
    #SCROLL_PATH = Path("/Users/forrest/dl.ash2txt.org/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/")
    SCROLL_PATH = Path("/Volumes/vesuvius/dl.ash2txt.org/data/full-scrolls/Scroll5/PHerc172.volpkg/volumes_zarr_standardized/53keV_7.91um_Scroll5.zarr/0")
    FIBER_SCROLL1 = Path("/Volumes/vesuvius/scroll1a_fibers/s1-surface-erode.zarr")
    #SCROLL_PATH = Path("/Volumes/vesuvius/dl.ash2txt.org/data/full-scrolls/Scroll1/PHercParis4.volpkg/volumes_zarr_standardized/54keV_7.91um_Scroll1A.zarr/0")
    CLUSTER_PATH = "/Volumes/vesuvius/scroll1a_clusters.zarr"
    SEGMENT_PATH = "/Volumes/vesuvius/scroll1a_segments.zarr"
    CHORD_PATH = "/Volumes/vesuvius/scroll1a_chords.zarr"

    OUTPUT_DIR = Path("output")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Compiling Code...")
    supervoxeler.compile_supervoxeler()
    snic.compile_snic('./c/snic.c','./libsnic.so')
    do_all_superpixeling(SCROLL_PATH, FIBER_SCROLL1, CLUSTER_PATH, CHORD_PATH, SEGMENT_PATH)

    # Visualize results
    print("Visualizing results...")
    #


if __name__ == "__main__":
    main()