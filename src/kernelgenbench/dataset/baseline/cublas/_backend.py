"""
cuBLAS baseline backend — delegates to kernelgenbench.runtime for multi-chip support.

Provides the same public API as before:
    get_or_create_handle, get_blas_func, map_op, map_fill_mode, map_side, map_diag
"""
import ctypes

try:
    from kernelgenbench.runtime import get_backend
except ImportError:
    # Standalone execution fallback — build a minimal shim
    get_backend = None

# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_lib_cache = None


def get_blas_lib():
    """Load and cache the BLAS shared library."""
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache
    backend = get_backend()
    _lib_cache = ctypes.CDLL(backend.blas_lib_path)
    return _lib_cache


# ---------------------------------------------------------------------------
# Handle management
# ---------------------------------------------------------------------------

_handle = None


def get_or_create_handle():
    """Get or create a global BLAS handle (reused across calls)."""
    global _handle
    if _handle is not None:
        return _handle

    lib = get_blas_lib()
    backend = get_backend()

    # Create handle
    create_fn = getattr(lib, backend.blas_create_handle_fn)
    create_fn.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    create_fn.restype = ctypes.c_int

    _handle = ctypes.c_void_p()
    status = create_fn(ctypes.byref(_handle))
    if status != 0:
        raise RuntimeError(f"BLAS handle creation failed with status {status}")

    # Set pointer mode to device
    spm_fn = getattr(lib, backend.blas_set_pointer_mode_fn)
    spm_fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
    spm_fn.restype = ctypes.c_int

    status = spm_fn(_handle, 1)  # 1 = DEVICE pointer mode
    if status != 0:
        raise RuntimeError(f"BLAS SetPointerMode failed with status {status}")

    return _handle


# ---------------------------------------------------------------------------
# Function loading
# ---------------------------------------------------------------------------


def get_blas_func(cublas_name: str, argtypes: list = None, restype=ctypes.c_int):
    """Load a BLAS function by its cuBLAS name, auto-mapped to the right backend."""
    lib = get_blas_lib()
    backend = get_backend()
    func_name = backend.map_blas_func_name(cublas_name)
    func = getattr(lib, func_name)
    if argtypes is not None:
        func.argtypes = argtypes
    func.restype = restype
    return func


# ---------------------------------------------------------------------------
# Enum mapping — thin wrappers around backend
# ---------------------------------------------------------------------------


def map_op(cublas_op) -> int:
    """Map cuBLAS operation enum to backend enum. Handles string input ('N'/'T'/'C')."""
    return get_backend().map_op(cublas_op)


def map_fill_mode(cublas_fill: int) -> int:
    return get_backend().map_fill_mode(cublas_fill)


def map_side(cublas_side: int) -> int:
    return get_backend().map_side(cublas_side)


def map_diag(cublas_diag: int) -> int:
    return get_backend().map_diag(cublas_diag)


def map_data_type(cublas_dtype: int) -> int:
    """Map cudaDataType enum to backend enum."""
    return get_backend().map_data_type(cublas_dtype)


# ---------------------------------------------------------------------------
# Pointer mode helper — some ops need to switch between HOST and DEVICE mode
# ---------------------------------------------------------------------------

_spm_fn = None


def set_pointer_mode(handle, mode: int):
    """Set BLAS pointer mode. 0 = HOST, 1 = DEVICE.

    On NVIDIA, this is a no-op (cuBLAS handles host/device pointers automatically).
    On other vendors (Hygon/Ascend/etc.), explicit mode switching is required.
    """
    backend = get_backend()
    # Skip pointer mode switching on NVIDIA — cuBLAS handles it automatically
    if backend.vendor.value == "nvidia":
        return

    global _spm_fn
    if _spm_fn is None:
        lib = get_blas_lib()
        _spm_fn = getattr(lib, backend.blas_set_pointer_mode_fn)
        _spm_fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _spm_fn.restype = ctypes.c_int
    status = _spm_fn(handle, mode)
    if status != 0:
        raise RuntimeError(f"BLAS SetPointerMode failed with status {status}")


# ---------------------------------------------------------------------------
# Complex number structs for ctypes interop
# ---------------------------------------------------------------------------


class cuComplex(ctypes.Structure):
    """cuBLAS complex64 struct (two floats)."""
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    """cuBLAS complex128 struct (two doubles)."""
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


# Alias used in some files with underscore prefix
_cuComplex = cuComplex
_cuDoubleComplex = cuDoubleComplex
