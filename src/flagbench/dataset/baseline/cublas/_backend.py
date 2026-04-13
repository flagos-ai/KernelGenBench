"""
Multi-chip BLAS backend abstraction.

Provides unified library loading, handle creation, and function name mapping
for cuBLAS (NVIDIA), hipBLAS (Hygon DCU), and future BLAS backends.
"""
import ctypes
import os
import functools

# ---------------------------------------------------------------------------
# Device detection (lightweight, no torch dependency)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _detect_backend() -> str:
    """Detect BLAS backend: 'cublas' or 'hipblas'."""
    vendor = os.environ.get("GEMS_VENDOR", "")
    if vendor == "hygon":
        return "hipblas"
    # Auto-detect via rocm-smi
    if not vendor:
        import subprocess
        try:
            r = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0 and any(
                k in r.stdout for k in ("Hygon", "DCU", "BW", "C-3000")
            ):
                return "hipblas"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return "cublas"


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

_lib_cache = None

def get_blas_lib():
    """Load and cache the BLAS shared library."""
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache

    backend = _detect_backend()
    if backend == "hipblas":
        # Hygon DCU: hipBLAS from DTK
        dtk_home = os.environ.get("DTK_HOME", "/opt/dtk-25.04")
        lib_path = os.path.join(dtk_home, "lib", "libhipblas.so")
        _lib_cache = ctypes.CDLL(lib_path)
    else:
        # NVIDIA: cuBLAS
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        _lib_cache = ctypes.CDLL(os.path.join(cuda_home, "lib64", "libcublas.so.12"))
    return _lib_cache


# ---------------------------------------------------------------------------
# Handle management
# ---------------------------------------------------------------------------

_handle = None
_set_pointer_mode = None

def get_or_create_handle():
    """Get or create a global BLAS handle (reused across calls)."""
    global _handle, _set_pointer_mode
    if _handle is not None:
        return _handle

    lib = get_blas_lib()
    backend = _detect_backend()

    # Create handle
    if backend == "hipblas":
        create_fn = lib.hipblasCreate
    else:
        create_fn = lib.cublasCreate_v2
    create_fn.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    create_fn.restype = ctypes.c_int

    _handle = ctypes.c_void_p()
    status = create_fn(ctypes.byref(_handle))
    if status != 0:
        raise RuntimeError(f"BLAS handle creation failed with status {status}")

    # Set pointer mode to device
    if backend == "hipblas":
        spm_fn = lib.hipblasSetPointerMode
    else:
        spm_fn = lib.cublasSetPointerMode_v2
    spm_fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
    spm_fn.restype = ctypes.c_int

    status = spm_fn(_handle, 1)  # 1 = DEVICE pointer mode
    if status != 0:
        raise RuntimeError(f"BLAS SetPointerMode failed with status {status}")

    _set_pointer_mode = spm_fn
    return _handle


# ---------------------------------------------------------------------------
# Function name mapping: cuBLAS -> hipBLAS
# ---------------------------------------------------------------------------

# hipBLAS naming convention:
#   - Most _v2 suffixes are dropped (cublasSgemm_v2 -> hipblasSgemm)
#   - Complex type _v2 functions keep _v2 (cublasCcopy_v2 -> hipblasCcopy_v2)
#   - _64 suffixes are dropped (cublasSgemmBatched_64 -> hipblasSgemmBatched)

# Explicit overrides for functions that KEEP _v2 in hipBLAS
_HIPBLAS_KEEP_V2 = {
    "cublasCcopy_v2", "cublasCdotu_v2", "cublasCgemm_v2", "cublasCgemv_v2",
    "cublasCgeru_v2", "cublasCsymm_v2", "cublasCsymv_v2",
    "cublasZdotc_v2", "cublasZgerc_v2", "cublasZswap_v2",
}


def get_blas_func_name(cublas_name: str) -> str:
    """Map a cuBLAS function name to the correct backend function name."""
    backend = _detect_backend()
    if backend == "cublas":
        return cublas_name

    # hipBLAS mapping
    hip_name = cublas_name.replace("cublas", "hipblas", 1)

    # Drop _64 suffix
    if hip_name.endswith("_64"):
        hip_name = hip_name[:-3]

    # Drop _v2 unless in keep list
    if cublas_name not in _HIPBLAS_KEEP_V2 and hip_name.endswith("_v2"):
        hip_name = hip_name[:-3]

    return hip_name


def get_blas_func(cublas_name: str, argtypes: list, restype=ctypes.c_int):
    """Load a BLAS function by its cuBLAS name, auto-mapped to the right backend."""
    lib = get_blas_lib()
    func_name = get_blas_func_name(cublas_name)
    func = getattr(lib, func_name)
    func.argtypes = argtypes
    func.restype = restype
    return func


# ---------------------------------------------------------------------------
# Enum mapping: cuBLAS uses 0/1/2, hipBLAS uses 111/112/113
# ---------------------------------------------------------------------------

# cuBLAS: CUBLAS_OP_N=0, CUBLAS_OP_T=1, CUBLAS_OP_C=2
# hipBLAS: HIPBLAS_OP_N=111, HIPBLAS_OP_T=112, HIPBLAS_OP_C=113
_CUBLAS_TO_HIPBLAS_OP = {0: 111, 1: 112, 2: 113}

# cuBLAS: CUBLAS_FILL_MODE_LOWER=0, CUBLAS_FILL_MODE_UPPER=1
# hipBLAS: HIPBLAS_FILL_MODE_LOWER=121, HIPBLAS_FILL_MODE_UPPER=122
_CUBLAS_TO_HIPBLAS_FILL = {0: 121, 1: 122}

# cuBLAS: CUBLAS_SIDE_LEFT=0, CUBLAS_SIDE_RIGHT=1
# hipBLAS: HIPBLAS_SIDE_LEFT=141, HIPBLAS_SIDE_RIGHT=142
_CUBLAS_TO_HIPBLAS_SIDE = {0: 141, 1: 142}

# cuBLAS: CUBLAS_DIAG_NON_UNIT=0, CUBLAS_DIAG_UNIT=1
# hipBLAS: HIPBLAS_DIAG_NON_UNIT=131, HIPBLAS_DIAG_UNIT=132
_CUBLAS_TO_HIPBLAS_DIAG = {0: 131, 1: 132}


def map_op(cublas_op: int) -> int:
    """Map cuBLAS operation enum to backend enum. Also handles string input."""
    if isinstance(cublas_op, str):
        cublas_op = {'N': 0, 'T': 1, 'C': 2}[cublas_op]
    if _detect_backend() == "hipblas":
        return _CUBLAS_TO_HIPBLAS_OP.get(cublas_op, cublas_op)
    return cublas_op


def map_fill_mode(cublas_fill: int) -> int:
    """Map cuBLAS fill mode enum to backend enum."""
    if _detect_backend() == "hipblas":
        return _CUBLAS_TO_HIPBLAS_FILL.get(cublas_fill, cublas_fill)
    return cublas_fill


def map_side(cublas_side: int) -> int:
    """Map cuBLAS side enum to backend enum."""
    if _detect_backend() == "hipblas":
        return _CUBLAS_TO_HIPBLAS_SIDE.get(cublas_side, cublas_side)
    return cublas_side


def map_diag(cublas_diag: int) -> int:
    """Map cuBLAS diag enum to backend enum."""
    if _detect_backend() == "hipblas":
        return _CUBLAS_TO_HIPBLAS_DIAG.get(cublas_diag, cublas_diag)
    return cublas_diag
