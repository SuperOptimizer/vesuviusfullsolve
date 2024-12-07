import numpy as np
import ctypes
from pathlib import Path
import subprocess



D_SEED=2
SUPERPIXEL_MAX_NEIGHS = 56 * 2


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


def run_snic(volume):
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
        Array of Superpixel structures
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
    lib = ctypes.CDLL(str(Path(__file__).parent / "libsnic.so"))

    # Configure function signature
    lib.snic.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8),  # img (changed from float32 to uint8)
        np.ctypeslib.ndpointer(dtype=np.uint32),  # labels
        ctypes.POINTER(Superpixel)  # superpixels
    ]
    lib.snic.restype = ctypes.c_int

    # Prepare output arrays
    labels = np.zeros(volume.shape, dtype=np.uint32)
    max_superpixels = snic_superpixel_count(lx, ly, lz) + 1
    superpixels = (Superpixel * max_superpixels)()

    # Run SNIC
    neigh_overflow = lib.snic(
        volume, labels, superpixels
    )

    if neigh_overflow > 0:
        print(f"Warning: {neigh_overflow} neighbor relationships exceeded storage capacity")

    return labels, superpixels


def snic_superpixel_count(lx, ly, lz):
    """Calculate the expected number of superpixels given dimensions and seed spacing."""
    return (lx // D_SEED) * (ly // D_SEED) * (lz // D_SEED)