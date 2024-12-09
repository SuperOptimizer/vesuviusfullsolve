import numpy as np
import ctypes
from pathlib import Path
import subprocess
import warnings
from typing import Tuple, List, Optional

# Constants
SUPERPIXEL_MAX_NEIGHS = 27


class Superpixel_ctype(ctypes.Structure):
    """C-compatible structure for superpixel data."""
    _fields_ = [
        ("x", ctypes.c_float),
        ("y", ctypes.c_float),
        ("z", ctypes.c_float),
        ("c", ctypes.c_float),
        ("n", ctypes.c_uint32),
        ("neighs", ctypes.c_uint32 * SUPERPIXEL_MAX_NEIGHS)
    ]


class Superpixel:
    """Python class representing a superpixel with its properties."""

    def __init__(self, z: float, y: float, x: float, c: float, n: int, neighs: List[int]):
        self.z = z  # z-coordinate of centroid
        self.y = y  # y-coordinate of centroid
        self.x = x  # x-coordinate of centroid
        self.c = c  # average intensity
        self.n = n  # number of pixels
        self.neighs = [n for n in neighs if n > 0]  # non-zero neighbors

    def __repr__(self) -> str:
        return f"Superpixel(pos=({self.z:.1f}, {self.y:.1f}, {self.x:.1f}), intensity={self.c:.1f}, size={self.n})"


class SNICError(Exception):
    """Custom exception for SNIC-related errors."""
    pass


def _load_snic_library(lib_path: Optional[Path] = None) -> ctypes.CDLL:
    """Load the SNIC shared library, compiling it if necessary."""
    if lib_path is None:
        lib_path = Path(__file__).parent / "libsnic.so"

    if not lib_path.exists():
        source_path = Path(__file__).parent / "c/snic.c"
        if not source_path.exists():
            raise SNICError(f"Source file not found: {source_path}")

        try:
            compile_snic(source_path, lib_path)
        except subprocess.CalledProcessError as e:
            raise SNICError(f"Failed to compile SNIC library: {e}")

    try:
        return ctypes.CDLL(str(lib_path))
    except OSError as e:
        raise SNICError(f"Failed to load SNIC library: {e}")


def compile_snic(source_path: Path, lib_path: Path) -> None:
    """Compile the SNIC C code into a shared library."""
    compile_cmd = [
        "gcc", "-O3", "-march=native", "-ffast-math",
        "-shared", "-fPIC",
        f"-DSUPERPIXEL_MAX_NEIGHS={SUPERPIXEL_MAX_NEIGHS}",
        str(source_path), "-o", str(lib_path)
    ]

    subprocess.run(compile_cmd, check=True)


def run_snic(
        volume: np.ndarray,
        d_seed: int,
        compactness: float = 40.0
) -> Tuple[np.ndarray, List[Superpixel], int]:
    """
    Run SNIC superpixel segmentation on a 3D volume.

    Parameters
    ----------
    volume : np.ndarray
        Input 3D volume (uint8 or float in range [0, 1])
    d_seed : int
        Spacing between superpixel seeds (must be a power of 2)
    compactness : float, optional
        Compactness factor (default: 40.0)

    Returns
    -------
    labels : np.ndarray
        Integer array of superpixel labels
    superpixels : List[Superpixel]
        List of Superpixel objects containing centroid, intensity, and neighbor information
    neigh_overflow : int
        Number of neighbors that couldn't be added due to reaching SUPERPIXEL_MAX_NEIGHS

    Raises
    ------
    SNICError
        If input validation fails or SNIC execution fails
    """
    # Input validation and normalization
    if volume.ndim != 3:
        raise SNICError(f"Expected 3D volume, got {volume.ndim}D")

    if not volume.flags['C_CONTIGUOUS']:
        volume = np.ascontiguousarray(volume)

    # Handle float inputs in [0, 1]
    if volume.dtype == np.float32 or volume.dtype == np.float64:
        if volume.min() < 0 or volume.max() > 1:
            raise SNICError("Float volume must be in range [0, 1]")
        volume = (volume * 255).astype(np.uint8)
    elif volume.dtype != np.uint8:
        raise SNICError(f"Unsupported dtype: {volume.dtype}, expected uint8 or float")

    # Validate dimensions
    lz, ly, lx = volume.shape
    max_dim = max(lz, ly, lx)
    if max_dim >= 2 ** 16:
        raise SNICError(f"Volume dimension {max_dim} exceeds uint16 limit")

    def is_power_of_2(n: int) -> bool:
        return n > 0 and (n & (n - 1)) == 0

    if not is_power_of_2(d_seed):
        raise SNICError(f"d_seed must be a power of 2, got {d_seed}")

    if d_seed >= min(volume.shape):
        raise SNICError(f"d_seed ({d_seed}) must be smaller than smallest dimension ({min(volume.shape)})")

    # Load and configure library
    lib = _load_snic_library()
    lib.snic.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8),
        ctypes.c_uint16,
        ctypes.c_uint16,
        ctypes.c_uint16,
        ctypes.c_uint32,
        ctypes.c_float,
        np.ctypeslib.ndpointer(dtype=np.uint32),
        ctypes.POINTER(Superpixel_ctype)
    ]
    lib.snic.restype = ctypes.c_int

    # Prepare outputs
    labels = np.zeros(volume.shape, dtype=np.uint32)
    max_superpixels = ((lz // d_seed) * (ly // d_seed) * (lx // d_seed)) + 1
    superpixels = (Superpixel_ctype * int(max_superpixels))()

    # Run SNIC
    neigh_overflow = lib.snic(
        volume,
        lz, ly, lx,
        d_seed,
        compactness,
        labels,
        superpixels
    )

    if neigh_overflow > 0:
        warnings.warn(f"Some superpixels exceeded maximum neighbor count ({SUPERPIXEL_MAX_NEIGHS})")

    # Convert C superpixels to Python objects
    superpixel_list = []
    empty_count = 0
    for i in range(1, max_superpixels):  # Skip index 0
        sp = superpixels[i]
        if sp.n > 0 and sp.c >= 0:  # Valid superpixel
            superpixel_list.append(Superpixel(
                sp.z, sp.y, sp.x, sp.c, sp.n,
                [n for n in sp.neighs]
            ))
        else:
            empty_count += 1

    if empty_count > 0:
        warnings.warn(f"Found {empty_count} empty superpixels")

    return labels, superpixel_list, neigh_overflow


def estimate_superpixel_count(volume_shape: Tuple[int, int, int], d_seed: int) -> int:
    """
    Calculate the expected number of superpixels given dimensions and seed spacing.

    Parameters
    ----------
    volume_shape : Tuple[int, int, int]
        Shape of the input volume (z, y, x)
    d_seed : int
        Spacing between superpixel seeds

    Returns
    -------
    int
        Expected number of superpixels
    """
    lz, ly, lx = volume_shape
    return (lz // d_seed) * (ly // d_seed) * (lx // d_seed)