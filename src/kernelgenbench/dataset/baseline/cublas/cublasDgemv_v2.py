import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_op, set_pointer_mode
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_op, set_pointer_mode

# cuBLAS operation types
CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
CUBLAS_OP_C = 2

# Global variables for caching (initialized once, reused)
_cublas_func = None


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasDgemv_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # trans
            ctypes.c_int,  # m
            ctypes.POINTER(ctypes.c_double),  # n
            ctypes.POINTER(ctypes.c_double),  # alpha
            ctypes.c_int,  # A
            ctypes.POINTER(ctypes.c_double),  # lda
            ctypes.c_int,  # x
            ctypes.POINTER(ctypes.c_double),  # incx
            ctypes.POINTER(ctypes.c_double),  # beta
            ctypes.c_int,  # y
        ])
    return _cublas_func




def cublasDgemv_v2(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    handle = get_or_create_handle()
    func = _get_cublas_func()

    # Convert trans
    if isinstance(trans, str):
        trans_map = {'N': CUBLAS_OP_N, 'T': CUBLAS_OP_T, 'C': CUBLAS_OP_C}
        trans = trans_map[trans]
    trans = map_op(trans)

    # Convert scalars
    alpha_c = ctypes.c_double(float(alpha))
    beta_c = ctypes.c_double(float(beta))

    # Get pointers
    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(ctypes.c_double))
    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(ctypes.c_double))
    y_ptr = ctypes.cast(y.data_ptr(), ctypes.POINTER(ctypes.c_double))

    # Switch to HOST pointer mode for alpha/beta
    set_pointer_mode(handle, 0)  # HOST

    # Call cuBLAS
    status = func(
        handle, trans, m, n,
        ctypes.byref(alpha_c),
        A_ptr, lda,
        x_ptr, incx,
        ctypes.byref(beta_c),
        y_ptr, incy
    )

    # Restore DEVICE pointer mode
    set_pointer_mode(handle, 1)  # DEVICE

    if status != 0:
        raise RuntimeError(f"cublasDgemv_v2 failed with status {status}")

    return y
