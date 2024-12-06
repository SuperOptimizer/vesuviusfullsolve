import numpy as np
import ctypes
from pathlib import Path
import subprocess




class Superpixel(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("c", ctypes.c_float),
        ("n", ctypes.c_uint32),
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


def run_snic(volume, d_seed, compactness=40.0):
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
        np.ctypeslib.ndpointer(dtype=np.uint32),  # labels
        ctypes.POINTER(Superpixel)  # superpixels
    ]
    lib.snic.restype = None  # Function now returns void

    # Prepare output arrays
    labels = np.zeros(volume.shape, dtype=np.uint32)
    max_superpixels = snic_superpixel_count(lx, ly, lz, d_seed) + 1
    superpixels = (Superpixel * max_superpixels)()

    # Run SNIC
    lib.snic(
        volume, lz, ly, lx, d_seed, compactness, labels, superpixels
    )

    return labels, superpixels


def snic_superpixel_count(lx, ly, lz, d_seed):
    """Calculate the expected number of superpixels given dimensions and seed spacing."""
    return (lx // d_seed) * (ly // d_seed) * (lz // d_seed)