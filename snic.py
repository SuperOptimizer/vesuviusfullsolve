import numpy as np
import ctypes
from pathlib import Path
import subprocess
class Superpixel(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_uint8),
        ("y", ctypes.c_uint8),
        ("z", ctypes.c_uint8),
        ("c", ctypes.c_uint8),  # Changed from c_float to c_uint8
        ("n", ctypes.c_uint32),
    ]

def run_snic(volume, d_seed, compactness=40.0):
    """
    Run SNIC superpixel segmentation on a 3D volume.

    Parameters:
    -----------
    volume : ndarray
        Input 3D volume as uint8 with values in [0,255]
    d_seed : int
        Seed spacing (controls number of superpixels)
    compactness : float
        Spatial regularization weight

    Returns:
    --------
    labels : ndarray
        Integer array of superpixel labels
    superpixels : ctypes array
        Array of Superpixel structures with uint8 coordinates and intensities
    """
    if not volume.flags['C_CONTIGUOUS']:
        volume = np.ascontiguousarray(volume)

    if volume.dtype != np.uint8:
        volume = (volume * 255).clip(0, 255).astype(np.uint8)

    if volume.shape != (256, 256, 256):
        raise ValueError("Volume must be exactly 256 x 256 x 256")

    # Load the compiled library
    lib = ctypes.CDLL(str(Path(__file__).parent / "libsnic.so"))

    # Configure function signature
    lib.snic.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8),  # Changed from float32 to uint8
        ctypes.c_int,
        ctypes.c_float,
        np.ctypeslib.ndpointer(dtype=np.uint32),
        ctypes.POINTER(Superpixel)
    ]
    lib.snic.restype = None

    # Prepare output arrays
    labels = np.zeros(volume.shape, dtype=np.uint32)
    max_superpixels = snic_superpixel_count(d_seed) + 1
    superpixels = (Superpixel * max_superpixels)()

    # Run SNIC
    lib.snic(volume, d_seed, compactness, labels, superpixels)

    return labels, superpixels

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


def snic_superpixel_count(d_seed):
    """Calculate the expected number of superpixels given seed spacing."""
    return (256 // d_seed) ** 3  # Simplified since we know dimensions are 256³