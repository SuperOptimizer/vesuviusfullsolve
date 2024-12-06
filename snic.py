import numpy as np
import ctypes
from pathlib import Path
import subprocess


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


def run_snic(volume, d_seed, compactness=40.0, lowmid=0.3, midhig=0.7):
    """
    Run SNIC superpixel segmentation on a 3D volume.

    Parameters:
    -----------
    volume : ndarray
        Input 3D volume as float32 with values in [0,1]
    d_seed : int
        Seed spacing (controls number of superpixels)
    compactness : float
        Spatial regularization weight
    lowmid : float
        Threshold between low and mid intensities (0-1)
    midhig : float
        Threshold between mid and high intensities (0-1)

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
        ctypes.c_int,  # lz
        ctypes.c_int,  # ly
        ctypes.c_int,  # lx
        ctypes.c_int,  # d_seed
        ctypes.c_float,  # compactness
        ctypes.c_float,  # lowmid
        ctypes.c_float,  # midhig
        np.ctypeslib.ndpointer(dtype=np.uint32),  # labels
        ctypes.POINTER(Superpixel)  # superpixels
    ]
    lib.snic.restype = ctypes.c_int

    # Prepare output arrays
    labels = np.zeros(volume.shape, dtype=np.uint32)
    max_superpixels = snic_superpixel_count(lx, ly, lz, d_seed) + 1
    superpixels = (Superpixel * max_superpixels)()

    # Run SNIC
    neigh_overflow = lib.snic(
        volume, lz, ly, lx, d_seed, compactness,
        lowmid, midhig, labels, superpixels
    )

    if neigh_overflow > 0:
        print(f"Warning: {neigh_overflow} neighbor relationships exceeded storage capacity")

    return labels, superpixels


def snic_superpixel_count(lx, ly, lz, d_seed):
    """Calculate the expected number of superpixels given dimensions and seed spacing."""
    return (lx // d_seed) * (ly // d_seed) * (lz // d_seed)