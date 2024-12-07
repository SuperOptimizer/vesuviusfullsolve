import numpy as np
import ctypes
from pathlib import Path
import subprocess

D_SEED= 2
DIMENSION = 256
COMPACTNESS = 100000.0

# Match the C struct definition exactly
class HeapNode(ctypes.Structure):
    _fields_ = [
        ("d", ctypes.c_float),
        ("k", ctypes.c_uint32),
        ("x", ctypes.c_uint16),
        ("y", ctypes.c_uint16),
        ("z", ctypes.c_uint16),
        ("pad", ctypes.c_uint16)
    ]


SUPERPIXEL_MAX_NEIGHS = 56 * 2


class Superpixel(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("c", ctypes.c_float),
        ("n", ctypes.c_uint32),
        ("nlow", ctypes.c_uint32),
        ("nmid", ctypes.c_uint32),
        ("nhig", ctypes.c_uint32),
        ("neighs", ctypes.c_uint32 * SUPERPIXEL_MAX_NEIGHS)
    ]


def compile_snic():
    """Compile the SNIC C code into a shared library."""
    source_path = Path(__file__).parent / "c/snic.c"
    lib_path = Path(__file__).parent / "libsnic.so"

    compile_cmd = [
        "clang", "-O3", "-march=native", "-ffast-math",
        "-shared", "-fPIC", "-fopenmp",
        str(source_path), "-o", str(lib_path)
    ]

    subprocess.run(compile_cmd, check=True)
    return lib_path


def run_snic(volume):
    """
    Run SNIC superpixel segmentation on a 3D volume.

    Parameters:
    -----------
    volume : ndarray
        Input 3D volume as float32 with values in [0,1]

    Returns:
    --------
    labels : ndarray
        Integer array of superpixel labels
    superpixels : ctypes array
        Array of Superpixel structures
    """
    if not volume.flags['C_CONTIGUOUS']:
        volume = np.ascontiguousarray(volume)

    if volume.dtype != np.float32:
        volume = volume.astype(np.float32)

    lz, ly, lx = volume.shape

    # Load the compiled library
    lib = ctypes.CDLL(str(Path(__file__).parent / "libsnic.so"))

    # Configure function signature
    lib.snic.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float32),  # img
        np.ctypeslib.ndpointer(dtype=np.uint32),  # labels
        ctypes.POINTER(Superpixel)  # superpixels
    ]
    lib.snic.restype = ctypes.c_int

    # Prepare output arrays
    labels = np.zeros(volume.shape, dtype=np.uint32)
    max_superpixels = snic_superpixel_count() + 1
    superpixels = (Superpixel * int(max_superpixels))()

    # Run SNIC
    neigh_overflow = lib.snic(
        volume,  labels, superpixels
    )

    if neigh_overflow > 0:
        print(f"Warning: {neigh_overflow} neighbor relationships exceeded storage capacity")

    return labels, superpixels


def snic_superpixel_count():
    """Calculate the expected number of superpixels given dimensions and seed spacing."""
    return (DIMENSION // D_SEED) * (DIMENSION // D_SEED) * (DIMENSION // D_SEED)