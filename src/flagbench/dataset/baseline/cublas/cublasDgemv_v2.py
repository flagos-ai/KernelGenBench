import torch
import ctypes
import os
import atexit

# Global variables for caching (initialized once, reused)
_libcublas = None
_cublas_handle = None
_cublas_func = None

# cuBLAS operation types
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2


def _get_cublas_lib():
    global _libcublas
    if _libcublas is None:
        cuda_home = os.environ.get('CUDA_HOME', '/usr/local/cuda')
        _libcublas = ctypes.CDLL(os.path.join(cuda_home, 'lib64', 'libcublas.so.12'))
    return _libcublas


def _get_or_create_handle():
    global _cublas_handle
    if _cublas_handle is None:
        libcublas = _get_cublas_lib()
        cublasCreate_v2 = libcublas.cublasCreate_v2
        cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cublasCreate_v2.restype = ctypes.c_int
        _cublas_handle = ctypes.c_void_p()
        status = cublasCreate_v2(ctypes.byref(_cublas_handle))
        if status != 0:
            raise RuntimeError(f"cublasCreate_v2 failed with status {status}")
    return _cublas_handle


def _get_cublas_func():
    global _cublas_func
    if _cublas_func is None:
        libcublas = _get_cublas_lib()
        _cublas_func = libcublas.cublasDgemv_v2
        _cublas_func.argtypes = [
            ctypes.c_void_p,                 # handle
            ctypes.c_int,                    # trans
            ctypes.c_int,                    # m
            ctypes.c_int,                    # n
            ctypes.POINTER(ctypes.c_double), # alpha
            ctypes.POINTER(ctypes.c_double), # A
            ctypes.c_int,                    # lda
            ctypes.POINTER(ctypes.c_double), # x
            ctypes.c_int,                    # incx
            ctypes.POINTER(ctypes.c_double), # beta
            ctypes.POINTER(ctypes.c_double), # y
            ctypes.c_int,                    # incy
        ]
        _cublas_func.restype = ctypes.c_int
    return _cublas_func


def cleanup_cublas():
    global _cublas_handle
    if _cublas_handle is not None:
        libcublas = _get_cublas_lib()
        libcublas.cublasDestroy_v2(_cublas_handle)
        _cublas_handle = None


atexit.register(cleanup_cublas)


def cublasDgemv_v2(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    handle = _get_or_create_handle()
    func = _get_cublas_func()

    # Convert trans
    if isinstance(trans, str):
        trans_map = {'N': CUBLAS_OP_N, 'T': CUBLAS_OP_T, 'C': CUBLAS_OP_C}
        trans = trans_map[trans]

    # Convert scalars
    alpha_c = ctypes.c_double(float(alpha))
    beta_c = ctypes.c_double(float(beta))

    # Get pointers
    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_double))
    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_double))
    y_ptr = ctypes.cast(y.data_ptr(), ctypes.POINTER(ctypes.c_double))

    # Call cuBLAS
    status = func(
        handle, trans, m, n,
        ctypes.byref(alpha_c),
        A_ptr, lda,
        x_ptr, incx,
        ctypes.byref(beta_c),
        y_ptr, incy
    )

    if status != 0:
        raise RuntimeError(f"cublasDgemv_v2 failed with status {status}")

    return y
