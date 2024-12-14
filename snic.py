import numpy as np
import ctypes
from pathlib import Path
import subprocess
import warnings
from typing import Tuple, List, Optional

class Superpixel_ctype(ctypes.Structure):
    """C-compatible structure for superpixel data."""
    _fields_ = [
        ("z", ctypes.c_float),
        ("y", ctypes.c_float),
        ("x", ctypes.c_float),
        ("c", ctypes.c_float),
        ("n", ctypes.c_uint32),
    ]

class Superpixel:
    """Python class representing a superpixel with its properties."""

    def __init__(self, z: float, y: float, x: float, c: float, n: int, connections: dict = None):
        self.z = z
        self.y = y
        self.x = x
        self.c = c
        self.n = n
        self.connections = connections or {}  # Dict mapping neighbor_label -> connection_strength

    def __repr__(self) -> str:
        return f"Superpixel(pos=({self.z:.1f}, {self.y:.1f}, {self.x:.1f}), intensity={self.c:.1f}, size={self.n}, neighbors={len(self.connections)})"

class SuperpixelConnection_ctype(ctypes.Structure):
    """C-compatible structure for connection data."""
    _fields_ = [
        ("neighbor_label", ctypes.c_uint32),
        ("connection_strength", ctypes.c_float),
    ]

class SuperpixelConnections_ctype(ctypes.Structure):
    """C-compatible structure for all connections of a superpixel."""
    _fields_ = [
        ("connections", ctypes.POINTER(SuperpixelConnection_ctype)),
        ("num_connections", ctypes.c_int),
    ]

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
        str(source_path), "-o", str(lib_path)
    ]

    subprocess.run(compile_cmd, check=True)


def run_snic(
        volume: np.ndarray,
        d_seed: int,
        compactness: float = 40.0, min_superpixel_size=1
) -> Tuple[np.ndarray, List[Superpixel]]:
    """Run SNIC superpixel segmentation on a 3D volume."""
    # ... (previous input validation code remains the same) ...

    lz,ly,lx = volume.shape

    # Load and configure SNIC library
    lib = _load_snic_library()

    # Configure SNIC function
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

    # Configure connection calculation function
    lib.calculate_superpixel_connections.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.uint8),
        np.ctypeslib.ndpointer(dtype=np.uint32),
        ctypes.c_uint16,
        ctypes.c_uint16,
        ctypes.c_uint16,
        ctypes.c_uint32
    ]
    lib.calculate_superpixel_connections.restype = ctypes.POINTER(SuperpixelConnections_ctype)

    # Prepare outputs
    labels = np.zeros(volume.shape, dtype=np.uint32)
    max_superpixels = ((lz // d_seed) * (ly // d_seed) * (lx // d_seed)) + 1
    superpixels = (Superpixel_ctype * int(max_superpixels))()

    # Run SNIC
    result = lib.snic(
        volume,
        lz, ly, lx,
        d_seed,
        compactness,
        labels,
        superpixels
    )

    if result != 0:
        raise SNICError("SNIC execution failed")

    # Calculate connections
    connections = lib.calculate_superpixel_connections(
        volume,
        labels,
        lz, ly, lx,
        max_superpixels - 1
    )

    # First pass: Create all superpixel objects without connections
    superpixel_list = []
    # Map from C label -> Python Superpixel object
    label_to_superpixel = {}

    max_connections=64

    empty_count = 0
    for i in range(1, max_superpixels):  # Skip index 0
        sp = superpixels[i]
        if sp.n >= min_superpixel_size and sp.c > 0:  # Valid superpixel
            new_superpixel = Superpixel(
                sp.z, sp.y, sp.x, sp.c, sp.n,
                connections={}
            )
            superpixel_list.append(new_superpixel)
            label_to_superpixel[i] = new_superpixel
        else:
            empty_count += 1

    # Second pass: Add connections using Superpixel object references
    # Modified second pass to keep top N connections
    for i in range(1, max_superpixels):
        if i in label_to_superpixel:  # If this is a valid superpixel
            current_superpixel = label_to_superpixel[i]

            # Collect all valid connections first
            all_connections = []
            if connections[i].num_connections > 0:
                for j in range(connections[i].num_connections):
                    conn = connections[i].connections[j]
                    neighbor_label = conn.neighbor_label

                    if neighbor_label in label_to_superpixel:
                        neighbor_superpixel = label_to_superpixel[neighbor_label]
                        all_connections.append((neighbor_superpixel, conn.connection_strength))

            # Sort connections by strength (highest to lowest) and keep top N
            sorted_connections = sorted(all_connections, key=lambda x: x[1], reverse=True)
            top_connections = sorted_connections[:max_connections]

            # Store only the top N connections
            current_superpixel.connections = {sp: strength for sp, strength in top_connections}

    # Free C memory
    lib.free_superpixel_connections(connections, max_superpixels - 1)

    if empty_count > 0:
        warnings.warn(f"Found {empty_count} empty superpixels")

    return labels, superpixel_list


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