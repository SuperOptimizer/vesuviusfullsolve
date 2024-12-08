import numpy as np
import ctypes
from pathlib import Path
import subprocess

D_SEED = 2
GRID_DIM = 256 // D_SEED


class Supervoxel(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("c", ctypes.c_float),
        ("n", ctypes.c_uint32),
    ]


class SupervoxelGrid(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(Supervoxel)),
    ]


def compile_supervoxeler():
    """Compile the Supervoxeler C code into a shared library."""
    source_path = Path(__file__).parent / "c/supervoxeler.c"
    lib_path = Path(__file__).parent / "libsupervoxeler.so"

    compile_cmd = [
        "clang", "-O3", "-march=native", "-ffast-math",
        "-shared", "-fPIC", "-fopenmp",
        str(source_path), "-o", str(lib_path)
    ]

    subprocess.run(compile_cmd, check=True)
    return lib_path


def run_supervoxeler(volume):
    """
    Run Supervoxeler supervoxel creation on a 3D volume.

    Parameters:
    -----------
    volume : ndarray
        Input 3D volume as uint8 with values in [0,255]

    Returns:
    --------
    ndarray
        3D structured array of supervoxels with shape (GRID_DIM, GRID_DIM, GRID_DIM)
        Each supervoxel contains (x, y, z, intensity, count)
    """
    if not volume.flags['C_CONTIGUOUS']:
        volume = np.ascontiguousarray(volume)

    if volume.dtype != np.uint8:
        if volume.dtype == np.float32 and volume.max() <= 1.0:
            volume = (volume * 255).astype(np.uint8)
        else:
            raise ValueError("Input volume must be uint8 or float32 in range [0,1]")

    lz, ly, lx = volume.shape
    if lx != 256 or ly != 256 or lz != 256:
        raise ValueError("Input volume must be 256x256x256")

    # Load the compiled library
    lib = ctypes.CDLL(str(Path(__file__).parent / "libsupervoxeler.so"))

    # Configure function signatures
    lib.init_supervoxel_grid.argtypes = None
    lib.init_supervoxel_grid.restype = ctypes.POINTER(SupervoxelGrid)

    lib.free_supervoxel_grid.argtypes = [ctypes.POINTER(SupervoxelGrid)]
    lib.free_supervoxel_grid.restype = None

    lib.supervoxeler.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8),
        ctypes.POINTER(SupervoxelGrid)
    ]
    lib.supervoxeler.restype = ctypes.c_int

    # Initialize grid
    grid = lib.init_supervoxel_grid()
    if not grid:
        raise RuntimeError("Failed to allocate supervoxel grid")

    try:
        # Run Supervoxeler
        num_supervoxels = lib.supervoxeler(volume, grid)
        if num_supervoxels < 0:
            raise RuntimeError("Supervoxeler algorithm failed")

        # Copy results into numpy array before freeing C memory
        result_shape = (GRID_DIM, GRID_DIM, GRID_DIM)
        supervoxels = np.zeros(result_shape, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('c', 'f4'), ('n', 'u4')
        ])

        # Copy the flat array into our 3D structured array
        flat_sv = np.ctypeslib.as_array(
            grid.contents.data,
            shape=(GRID_DIM * GRID_DIM * GRID_DIM,)
        )
        supervoxels.ravel()[:] = flat_sv

        return supervoxels

    finally:
        # Clean up C memory
        lib.free_supervoxel_grid(grid)


def get_supervoxel(supervoxels, grid_x, grid_y, grid_z):
    """
    Helper function to access supervoxel at specific grid coordinates.

    Parameters:
    -----------
    supervoxels : ndarray
        3D structured array output from run_supervoxeler
    grid_x, grid_y, grid_z : int
        Grid coordinates

    Returns:
    --------
    tuple:
        (x, y, z, intensity, count) for the supervoxel at those coordinates
    """
    if not (0 <= grid_x < GRID_DIM and
            0 <= grid_y < GRID_DIM and
            0 <= grid_z < GRID_DIM):
        return None

    sv = supervoxels[grid_z, grid_y, grid_x]
    return sv['x'], sv['y'], sv['z'], sv['c'], sv['n']


def volume_to_grid(vol_x, vol_y, vol_z):
    """Convert volume coordinates to grid coordinates."""
    return (vol_x // D_SEED, vol_y // D_SEED, vol_z // D_SEED)


def supervoxel_count():
    """Calculate the total number of supervoxels in the grid."""
    return GRID_DIM * GRID_DIM * GRID_DIM