"""CANN baseline backend — ctypes calls to CANN aclnn C API.

This module provides the infrastructure for calling CANN aclnn operators
via ctypes, completely bypassing PyTorch/torch_npu dispatch.

Pattern: ctypes → CANN aclnn C API (aclnnGemm, aclnnBatchMatMul, etc.)

Two-stage calling convention:
  1. aclnn<Op>GetWorkspaceSize(...) → workspace_size, executor
  2. aclnn<Op>(workspace, workspace_size, executor, stream)

Public API:
    get_acl_lib()           — load libascendcl.so / libopapi.so
    create_acl_tensor()     — create aclTensor from PyTorch tensor
    destroy_acl_tensor()    — destroy aclTensor
    create_acl_scalar()     — create aclScalar
    destroy_acl_scalar()    — destroy aclScalar
    create_acl_int_array()  — create aclIntArray
    destroy_acl_int_array() — destroy aclIntArray
    get_or_create_stream()  — get current NPU stream
    allocate_workspace()    — allocate device workspace memory
    free_workspace()        — free device workspace memory
    two_stage_launch()      — generic two-stage call wrapper
    map_transpose()         — 'N'/'T'/'C' or 0/1/2 → int
    map_conjugate()         — check if conjugate
    map_fill_upper()        — cuBLAS fill mode → bool
    map_side_left()         — cuBLAS side mode → bool
    map_unit_diagonal()     — cuBLAS diag type → bool
    torch_dtype_to_acl()    — torch.dtype → aclDataType enum
    cuda_dtype_to_torch()   — cudaDataType enum → torch.dtype

Reference: /home/zpy/cublas_vs_cann_signature_comparison.md
"""
import ctypes
import os
import torch

try:
    import torch_npu  # noqa: F401
    _NPU_AVAILABLE = True
except ImportError:
    _NPU_AVAILABLE = False

# =========================================================================
# aclDataType enum values (from aclnn headers)
# =========================================================================
ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_INT8 = 2
ACL_INT32 = 3
ACL_UINT8 = 4
ACL_INT16 = 6
ACL_UINT16 = 7
ACL_UINT32 = 8
ACL_INT64 = 9
ACL_UINT64 = 10
ACL_DOUBLE = 11
ACL_BOOL = 12
ACL_COMPLEX64 = 16
ACL_COMPLEX128 = 17
ACL_BF16 = 27

# aclFormat
ACL_FORMAT_ND = 2

# =========================================================================
# dtype mapping: torch.dtype → aclDataType
# =========================================================================
_TORCH_TO_ACL_DTYPE = {
    torch.float32:    ACL_FLOAT,
    torch.float16:    ACL_FLOAT16,
    torch.float64:    ACL_DOUBLE,
    torch.int8:       ACL_INT8,
    torch.int16:      ACL_INT16,
    torch.int32:      ACL_INT32,
    torch.int64:      ACL_INT64,
    torch.uint8:      ACL_UINT8,
    torch.bool:       ACL_BOOL,
    torch.complex64:  ACL_COMPLEX64,
    torch.complex128: ACL_COMPLEX128,
    torch.bfloat16:   ACL_BF16,
}


def torch_dtype_to_acl(dtype) -> int:
    """Map torch.dtype to aclDataType enum."""
    if dtype in _TORCH_TO_ACL_DTYPE:
        return _TORCH_TO_ACL_DTYPE[dtype]
    raise ValueError(f"Unsupported torch dtype for ACL: {dtype}")


# cudaDataType enum → torch.dtype (for cublasSgemmEx etc.)
_CUDA_DTYPE_MAP = {
    0: torch.float32,    # CUDA_R_32F
    1: torch.float64,    # CUDA_R_64F
    2: torch.float16,    # CUDA_R_16F
    4: torch.complex64,  # CUDA_C_32F
    5: torch.complex128, # CUDA_C_64F
    8: torch.complex64,  # CUDA_C_32F (alternate)
}


def cuda_dtype_to_torch(cuda_dtype: int):
    """Map cudaDataType enum to torch.dtype."""
    if cuda_dtype in _CUDA_DTYPE_MAP:
        return _CUDA_DTYPE_MAP[cuda_dtype]
    raise ValueError(f"Unsupported cudaDataType: {cuda_dtype}")


# =========================================================================
# Library loading
# =========================================================================
_acl_lib = None
_opapi_lib = None


def _find_cann_lib(name):
    """Search for a CANN shared library."""
    # Try ASCEND_HOME_PATH first
    ascend_home = os.environ.get('ASCEND_HOME_PATH', '')
    if ascend_home:
        candidates = [
            os.path.join(ascend_home, 'lib64', name),
            os.path.join(ascend_home, 'acllib', 'lib64', name),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p

    # Try common install paths
    for base in [
        '/usr/local/Ascend/ascend-toolkit/latest',
        '/usr/local/Ascend/latest',
        '/usr/local/Ascend',
    ]:
        for sub in ['lib64', 'acllib/lib64', 'aarch64-linux/lib64',
                     'x86_64-linux/lib64']:
            p = os.path.join(base, sub, name)
            if os.path.isfile(p):
                return p

    # Fallback: let ctypes search LD_LIBRARY_PATH
    return name


def get_acl_lib():
    """Load and cache libascendcl.so (ACL runtime)."""
    global _acl_lib
    if _acl_lib is not None:
        return _acl_lib
    path = _find_cann_lib('libascendcl.so')
    _acl_lib = ctypes.CDLL(path)
    return _acl_lib


def get_opapi_lib():
    """Load and cache libopapi.so (aclnn operator API)."""
    global _opapi_lib
    if _opapi_lib is not None:
        return _opapi_lib
    path = _find_cann_lib('libopapi.so')
    _opapi_lib = ctypes.CDLL(path)
    return _opapi_lib


# =========================================================================
# Stream management
# =========================================================================
_stream = None


def get_or_create_stream():
    """Get the current NPU stream as a c_void_p for ctypes calls."""
    global _stream
    if _stream is not None:
        return _stream
    if _NPU_AVAILABLE and torch.npu.is_available():
        # torch_npu exposes current_stream().npu_stream as the raw stream ptr
        s = torch.npu.current_stream()
        # The underlying C pointer
        if hasattr(s, 'npu_stream'):
            _stream = ctypes.c_void_p(s.npu_stream)
        else:
            # Fallback: create stream via ACL C API
            _stream = _create_stream_via_acl()
    else:
        _stream = _create_stream_via_acl()
    return _stream


def _create_stream_via_acl():
    """Create a stream using aclrtCreateStream."""
    lib = get_acl_lib()
    stream = ctypes.c_void_p()
    fn = lib.aclrtCreateStream
    fn.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    fn.restype = ctypes.c_int
    status = fn(ctypes.byref(stream))
    if status != 0:
        raise RuntimeError(f"aclrtCreateStream failed with status {status}")
    return stream


# =========================================================================
# aclTensor creation / destruction
# =========================================================================
_create_tensor_fn = None
_destroy_tensor_fn = None


def create_acl_tensor(tensor, shape=None, stride=None, offset=0, dtype=None,
                      fmt=ACL_FORMAT_ND):
    """Create an aclTensor from a PyTorch tensor.

    Parameters:
        tensor: PyTorch tensor (provides data_ptr)
        shape:  list/tuple of ints (default: tensor.shape)
        stride: list/tuple of ints (default: tensor.stride())
        offset: storage offset in elements (default: 0)
        dtype:  aclDataType enum (default: auto from tensor.dtype)
        fmt:    aclFormat (default: ACL_FORMAT_ND)

    Returns:
        ctypes.c_void_p — opaque aclTensor pointer
    """
    global _create_tensor_fn
    if _create_tensor_fn is None:
        lib = get_opapi_lib()
        _create_tensor_fn = lib.aclCreateTensor
        # aclTensor* aclCreateTensor(
        #     const int64_t* viewDims, uint64_t viewDimsNum,
        #     aclDataType dataType, const int64_t* stride,
        #     int64_t offset, aclFormat format,
        #     const int64_t* storageDims, uint64_t storageDimsNum,
        #     void* data)
        _create_tensor_fn.argtypes = [
            ctypes.POINTER(ctypes.c_int64),  # viewDims
            ctypes.c_uint64,                 # viewDimsNum
            ctypes.c_int,                    # dataType
            ctypes.POINTER(ctypes.c_int64),  # stride
            ctypes.c_int64,                  # offset
            ctypes.c_int,                    # format
            ctypes.POINTER(ctypes.c_int64),  # storageDims
            ctypes.c_uint64,                 # storageDimsNum
            ctypes.c_void_p,                 # data
        ]
        _create_tensor_fn.restype = ctypes.c_void_p

    if shape is None:
        shape = list(tensor.shape)
    if stride is None:
        stride = list(tensor.stride())
    if dtype is None:
        dtype = torch_dtype_to_acl(tensor.dtype)

    ndim = len(shape)
    c_shape = (ctypes.c_int64 * ndim)(*shape)
    c_stride = (ctypes.c_int64 * ndim)(*stride)
    # storageDims = shape (for dense tensors)
    c_storage = (ctypes.c_int64 * ndim)(*shape)

    data_ptr = ctypes.c_void_p(tensor.data_ptr())

    acl_tensor = _create_tensor_fn(
        c_shape, ctypes.c_uint64(ndim),
        ctypes.c_int(dtype),
        c_stride,
        ctypes.c_int64(offset),
        ctypes.c_int(fmt),
        c_storage, ctypes.c_uint64(ndim),
        data_ptr,
    )
    if acl_tensor is None or acl_tensor == 0:
        raise RuntimeError("aclCreateTensor returned NULL")
    return ctypes.c_void_p(acl_tensor)


def destroy_acl_tensor(acl_tensor):
    """Destroy an aclTensor."""
    global _destroy_tensor_fn
    if acl_tensor is None or (isinstance(acl_tensor, ctypes.c_void_p) and acl_tensor.value is None):
        return
    if _destroy_tensor_fn is None:
        lib = get_opapi_lib()
        _destroy_tensor_fn = lib.aclDestroyTensor
        _destroy_tensor_fn.argtypes = [ctypes.c_void_p]
        _destroy_tensor_fn.restype = ctypes.c_int
    _destroy_tensor_fn(acl_tensor)


# =========================================================================
# aclScalar creation / destruction
# =========================================================================
_create_scalar_fn = None
_destroy_scalar_fn = None


def create_acl_scalar(value, dtype):
    """Create an aclScalar.

    Parameters:
        value: Python scalar (int, float, complex)
        dtype: aclDataType enum

    Returns:
        ctypes.c_void_p — opaque aclScalar pointer
    """
    global _create_scalar_fn
    if _create_scalar_fn is None:
        lib = get_opapi_lib()
        _create_scalar_fn = lib.aclCreateScalar
        # aclScalar* aclCreateScalar(void* data, aclDataType dataType)
        _create_scalar_fn.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _create_scalar_fn.restype = ctypes.c_void_p

    # Pack value into a ctypes buffer
    if dtype == ACL_FLOAT:
        buf = ctypes.c_float(float(value))
    elif dtype == ACL_DOUBLE:
        buf = ctypes.c_double(float(value))
    elif dtype == ACL_FLOAT16:
        # Pack as uint16 via numpy
        import numpy as np
        val_np = np.float16(value)
        buf = ctypes.c_uint16(val_np.view(np.uint16))
    elif dtype == ACL_INT32:
        buf = ctypes.c_int32(int(value))
    elif dtype == ACL_INT64:
        buf = ctypes.c_int64(int(value))
    elif dtype == ACL_COMPLEX64:
        # complex64 = two float32
        c = complex(value)

        class Complex64(ctypes.Structure):
            _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]
        buf = Complex64(c.real, c.imag)
    elif dtype == ACL_COMPLEX128:
        c = complex(value)

        class Complex128(ctypes.Structure):
            _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]
        buf = Complex128(c.real, c.imag)
    elif dtype == ACL_BOOL:
        buf = ctypes.c_bool(bool(value))
    else:
        # Fallback: try float
        buf = ctypes.c_float(float(value))

    acl_scalar = _create_scalar_fn(ctypes.byref(buf), ctypes.c_int(dtype))
    if acl_scalar is None or acl_scalar == 0:
        raise RuntimeError("aclCreateScalar returned NULL")
    return ctypes.c_void_p(acl_scalar)


def destroy_acl_scalar(acl_scalar):
    """Destroy an aclScalar."""
    global _destroy_scalar_fn
    if acl_scalar is None or (isinstance(acl_scalar, ctypes.c_void_p) and acl_scalar.value is None):
        return
    if _destroy_scalar_fn is None:
        lib = get_opapi_lib()
        _destroy_scalar_fn = lib.aclDestroyScalar
        _destroy_scalar_fn.argtypes = [ctypes.c_void_p]
        _destroy_scalar_fn.restype = ctypes.c_int
    _destroy_scalar_fn(acl_scalar)


# =========================================================================
# aclIntArray creation / destruction
# =========================================================================
_create_int_array_fn = None
_destroy_int_array_fn = None


def create_acl_int_array(values):
    """Create an aclIntArray from a list of ints."""
    global _create_int_array_fn
    if _create_int_array_fn is None:
        lib = get_opapi_lib()
        _create_int_array_fn = lib.aclCreateIntArray
        _create_int_array_fn.argtypes = [
            ctypes.POINTER(ctypes.c_int64),
            ctypes.c_uint64,
        ]
        _create_int_array_fn.restype = ctypes.c_void_p

    n = len(values)
    c_arr = (ctypes.c_int64 * n)(*values)
    result = _create_int_array_fn(c_arr, ctypes.c_uint64(n))
    if result is None or result == 0:
        raise RuntimeError("aclCreateIntArray returned NULL")
    return ctypes.c_void_p(result)


def destroy_acl_int_array(acl_arr):
    """Destroy an aclIntArray."""
    global _destroy_int_array_fn
    if acl_arr is None or (isinstance(acl_arr, ctypes.c_void_p) and acl_arr.value is None):
        return
    if _destroy_int_array_fn is None:
        lib = get_opapi_lib()
        _destroy_int_array_fn = lib.aclDestroyIntArray
        _destroy_int_array_fn.argtypes = [ctypes.c_void_p]
        _destroy_int_array_fn.restype = ctypes.c_int
    _destroy_int_array_fn(acl_arr)


# =========================================================================
# Workspace management
# =========================================================================
_malloc_fn = None
_free_fn = None


def allocate_workspace(size):
    """Allocate device memory for workspace.

    Returns ctypes.c_void_p (NULL if size == 0).
    """
    if size == 0:
        return ctypes.c_void_p(0)
    global _malloc_fn
    if _malloc_fn is None:
        lib = get_acl_lib()
        _malloc_fn = lib.aclrtMalloc
        _malloc_fn.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_int,  # aclrtMemMallocPolicy
        ]
        _malloc_fn.restype = ctypes.c_int
    ptr = ctypes.c_void_p()
    # policy = ACL_MEM_MALLOC_HUGE_FIRST (0)
    status = _malloc_fn(ctypes.byref(ptr), ctypes.c_size_t(size), ctypes.c_int(0))
    if status != 0:
        raise RuntimeError(f"aclrtMalloc failed with status {status} (size={size})")
    return ptr


def free_workspace(ptr):
    """Free device memory."""
    if ptr is None or (isinstance(ptr, ctypes.c_void_p) and (ptr.value is None or ptr.value == 0)):
        return
    global _free_fn
    if _free_fn is None:
        lib = get_acl_lib()
        _free_fn = lib.aclrtFree
        _free_fn.argtypes = [ctypes.c_void_p]
        _free_fn.restype = ctypes.c_int
    _free_fn(ptr)


# =========================================================================
# Two-stage call wrapper
# =========================================================================


def two_stage_launch(get_ws_func_name, exec_func_name, get_ws_args):
    """Generic two-stage aclnn call.

    Parameters:
        get_ws_func_name: str, e.g. 'aclnnGemmGetWorkspaceSize'
        exec_func_name:   str, e.g. 'aclnnGemm'
        get_ws_args:      list of ctypes-compatible arguments for GetWorkspaceSize
                          (excluding the trailing workspace_size and executor pointers)

    Returns:
        status code from the exec call (0 = success)
    """
    lib = get_opapi_lib()
    stream = get_or_create_stream()

    # --- Stage 1: GetWorkspaceSize ---
    get_ws_fn = getattr(lib, get_ws_func_name)
    workspace_size = ctypes.c_uint64(0)
    executor = ctypes.c_void_p(None)

    all_args = list(get_ws_args) + [
        ctypes.byref(workspace_size),
        ctypes.byref(executor),
    ]
    status = get_ws_fn(*all_args)
    if status != 0:
        raise RuntimeError(f"{get_ws_func_name} failed with status {status}")

    # --- Stage 2: Allocate workspace + Execute ---
    ws_ptr = allocate_workspace(workspace_size.value)

    exec_fn = getattr(lib, exec_func_name)
    status = exec_fn(ws_ptr, workspace_size, executor, stream)

    # Free workspace
    free_workspace(ws_ptr)

    if status != 0:
        raise RuntimeError(f"{exec_func_name} failed with status {status}")
    return status


# =========================================================================
# Synchronization
# =========================================================================


def synchronize():
    """Synchronize the NPU device."""
    if _NPU_AVAILABLE and torch.npu.is_available():
        torch.npu.synchronize()
    else:
        lib = get_acl_lib()
        stream = get_or_create_stream()
        fn = lib.aclrtSynchronizeStream
        fn.argtypes = [ctypes.c_void_p]
        fn.restype = ctypes.c_int
        fn(stream)


# =========================================================================
# Enum mapping helpers (cuBLAS convention → CANN convention)
# =========================================================================


def map_transpose(op) -> int:
    """Map cuBLAS operation enum to transpose int.

    cuBLAS: 0=N, 1=T, 2=C  (or string 'N'/'T'/'C')
    Returns: 0 (no transpose) or 1 (transpose)
    """
    if isinstance(op, str):
        return 1 if op in ('T', 'C') else 0
    return 1 if op in (1, 2) else 0


def map_conjugate(op) -> bool:
    """Check if operation requires conjugation (CUBLAS_OP_C = 2 or 'C')."""
    if isinstance(op, str):
        return op == 'C'
    return op == 2


def map_fill_upper(uplo) -> bool:
    """Map cuBLAS fill mode to upper boolean.

    cuBLAS: 0=LOWER, 1=UPPER
    """
    return uplo == 1


def map_side_left(side) -> bool:
    """Map cuBLAS side mode. 0=LEFT, 1=RIGHT. Returns True if left."""
    return side == 0


def map_unit_diagonal(diag) -> bool:
    """Map cuBLAS diag type. 0=NON_UNIT, 1=UNIT. Returns True if unit."""
    return diag == 1


# =========================================================================
# Device helpers
# =========================================================================

_device = None


def get_device():
    """Get the default NPU device (cached)."""
    global _device
    if _device is not None:
        return _device
    if _NPU_AVAILABLE and torch.npu.is_available():
        _device = torch.device('npu')
    else:
        _device = torch.device('cpu')
    return _device


def ensure_npu(tensor):
    """Ensure tensor is on NPU device."""
    dev = get_device()
    if tensor.device.type == dev.type:
        return tensor
    return tensor.to(dev)


# =========================================================================
# Scalar cache — reuse aclScalar objects across calls
# =========================================================================
_scalar_cache = {}


def get_cached_acl_scalar(key, value, acl_dtype):
    """Get or create a cached aclScalar."""
    if isinstance(value, complex):
        cache_key = (key, acl_dtype, value.real, value.imag)
    else:
        cache_key = (key, acl_dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = create_acl_scalar(value, acl_dtype)
    return _scalar_cache[cache_key]
