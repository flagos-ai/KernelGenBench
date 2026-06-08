import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_op, cuComplex, set_pointer_mode
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_op, cuComplex, set_pointer_mode

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
        _cublas_func = get_blas_func('cublasCgemv_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # trans
            ctypes.c_int,  # m
            ctypes.POINTER(cuComplex),  # n
            ctypes.POINTER(cuComplex),  # alpha
            ctypes.c_int,  # A
            ctypes.POINTER(cuComplex),  # lda
            ctypes.c_int,  # x
            ctypes.POINTER(cuComplex),  # incx
            ctypes.POINTER(cuComplex),  # beta
            ctypes.c_int,  # y
        ])
    return _cublas_func


def _torch_complex_to_cucomplex(val):
    """Convert torch complex scalar to cuComplex"""
    c = cuComplex()
    c.x = val.real
    c.y = val.imag
    return c




def cublasCgemv_v2(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    handle = get_or_create_handle()
    func = _get_cublas_func()

    # Convert trans
    if isinstance(trans, str):
        trans_map = {'N': CUBLAS_OP_N, 'T': CUBLAS_OP_T, 'C': CUBLAS_OP_C}
        trans = trans_map[trans]
    trans = map_op(trans)

    # Convert scalars
    alpha_c = _torch_complex_to_cucomplex(alpha)
    beta_c = _torch_complex_to_cucomplex(beta)

    # Get pointers
    A_ptr = ctypes.cast(A.data_ptr(), ctypes.POINTER(cuComplex))
    x_ptr = ctypes.cast(x.data_ptr(), ctypes.POINTER(cuComplex))
    y_ptr = ctypes.cast(y.data_ptr(), ctypes.POINTER(cuComplex))

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
        raise RuntimeError(f"cublasCgemv_v2 failed with status {status}")

    return y
