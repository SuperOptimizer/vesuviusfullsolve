import zarr
import os
import numpy as np

from snic import Superpixel


def superclusters_to_numpy(superclusters, padding=(0, 0, 0)):
    num_points = 64 * 64 * 64  # Maximum points per chunk
    data = np.full((num_points, 5), np.nan, dtype=np.float16)

    for i, sc in enumerate(superclusters):
        if i >= num_points:
            print(f"Warning: More superclusters ({len(superclusters)}) than available slots ({num_points})")
            break

        data[i] = [
            sc.z - padding[0],
            sc.y - padding[1],
            sc.x - padding[2],
            sc.c,
            sc.n
        ]
    return data


def numpy_to_superclusters(data):
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