import zarr
import os
import numpy as np

from snic import Superpixel

#scroll 1 A
# z 14376,
# y 7888,
# x 8096
#assuming d_seed = 2 and chunk z = y = x = 128
class ClusterVolume:
    def __init__(self, zarr_path, mode, create=False):
        if create:
            if os.path.exists(zarr_path):
                raise Exception(f"can't create a zarr at {zarr_path} because that directory already exists")
            self.root = zarr.open(zarr_path, mode='w')

            self.root.zeros('centroids_zyxcn', shape=(113, 63, 64, 64 * 64 * 64, 5),
                            chunks=(1, 1, 1, 64 * 64 * 64, 5),
                            dtype=np.float16)
        else:
            self.root = zarr.open(zarr_path, mode=mode)

    def write_superclusters(self, superclusters, chunk_indices, padding=(0,0,0)):
        """
        Write supercluster data to the zarr array at the specified chunk location.

        Parameters:
        superclusters: List[Superpixel] - output from run_snic
        chunk_indices: tuple(int, int, int) - (z_idx, y_idx, x_idx) indices of the chunk
        """
        print(f"writing {len(superclusters)} superclusters to {chunk_indices}")
        z_idx, y_idx, x_idx = chunk_indices

        # Create array to hold the supercluster data
        num_points = 64 * 64 * 64  # Maximum points per chunk
        data = np.full((num_points, 5), np.nan, dtype=np.float16)

        # Fill data array with supercluster information
        for i, sc in enumerate(superclusters):
            if i >= num_points:
                print(f"Warning: More superclusters ({len(superclusters)}) than available slots ({num_points})")
                break

            # Pack z,y,x,c,n into the array
            data[i] = [
                sc.z - padding[0],
                sc.y - padding[1],
                sc.x - padding[2],
                sc.c,
                sc.n
            ]

        # Write to zarr array
        self.root['centroids_zyxcn'][z_idx, y_idx, x_idx] = data

    def read_superclusters(self, chunk_indices):
        """
        Read supercluster data from the specified chunk location.

        Parameters:
        chunk_indices: tuple(int, int, int) - (z_idx, y_idx, x_idx) indices of the chunk

        Returns:
        List[Superpixel] - list of superclusters in this chunk
        """
        z_idx, y_idx, x_idx = chunk_indices
        data = self.root['centroids_zyxcn'][z_idx, y_idx, x_idx]

        superclusters = []
        for point in data:
            # Skip nan entries
            if np.isnan(point[0]):
                continue

            # Create Superpixel object
            sc = Superpixel(
                z=float(point[0]),
                y=float(point[1]),
                x=float(point[2]),
                c=float(point[3]),
                n=int(point[4])
            )
            superclusters.append(sc)

        return superclusters