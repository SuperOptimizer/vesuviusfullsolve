import numpy as np
import ctypes
from pathlib import Path
import subprocess

D_SEED = 2

class Supervoxel(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("c", ctypes.c_float),
        ("n", ctypes.c_uint32),
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
    supervoxels : ctypes array
        Array of Supervoxel structures containing the averaged positions and intensities
    num_supervoxels : int
        Number of supervoxels created
    """
    if not volume.flags['C_CONTIGUOUS']:
        volume = np.ascontiguousarray(volume)

    if volume.dtype != np.uint8:
        if volume.dtype == np.float32 and volume.max() <= 1.0:
            # Convert from float32 [0,1] to uint8 [0,255]
            volume = (volume * 255).astype(np.uint8)
        else:
            raise ValueError("Input volume must be uint8 or float32 in range [0,1]")

    lz, ly, lx = volume.shape

    # Load the compiled library
    lib = ctypes.CDLL(str(Path(__file__).parent / "libsupervoxeler.so"))

    # Configure function signature
    lib.supervoxeler.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8),  # img
        ctypes.POINTER(Supervoxel)  # supervoxels
    ]
    lib.supervoxeler.restype = ctypes.c_int

    # Prepare output arrays
    max_supervoxels = supervoxel_count(lx, ly, lz) + 1
    supervoxels = (Supervoxel * max_supervoxels)()

    # Run Supervoxeler
    num_supervoxels = lib.supervoxeler(volume, supervoxels)

    if num_supervoxels < 0:
        raise RuntimeError("Supervoxeler algorithm failed to allocate memory")

    return supervoxels, num_supervoxels

def supervoxel_count(lx, ly, lz):
    """Calculate the expected number of supervoxels given dimensions and seed spacing."""
    return (lx // D_SEED) * (ly // D_SEED) * (lz // D_SEED)