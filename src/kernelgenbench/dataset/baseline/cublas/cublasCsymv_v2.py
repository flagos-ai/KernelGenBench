import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_fill_mode, cuComplex
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_fill_mode, cuComplex

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasCsymv_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # uplo
            ctypes.POINTER(cuComplex),  # n
            ctypes.c_void_p,  # alpha (device pointer)
            ctypes.c_int,  # A (device pointer)
            ctypes.c_void_p,  # lda
            ctypes.c_int,  # x (device pointer)
            ctypes.POINTER(cuComplex),  # incx
            ctypes.c_void_p,  # beta (device pointer)
            ctypes.c_int,  # y (device pointer)
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    if dtype.is_complex:
        cache_key = (key, dtype, complex(value))
    else:
        cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasCsymv_v2(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    '''ctypes cuBLAS C API baseline for cublasCsymv_v2'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    uplo = map_fill_mode(uplo)
    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    # Get cached scalar GPU tensors for alpha and beta (complex64)
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.complex64)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.complex64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Call cuBLAS C API
    func(
        handle,
        uplo,
        n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuComplex)),
        A_ptr,
        lda,
        x_ptr,
        incx,
        ctypes.cast(beta_ptr, ctypes.POINTER(cuComplex)),
        y_ptr,
        incy
    )

    return y

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    n = 5
    # Create complex64 symmetric matrix A (A == A.t(), not Hermitian)
    M = torch.randn(n, n, device='cuda', dtype=torch.complex64)
    A = 0.5 * (M + M.t())
    x = torch.randn(n, device='cuda', dtype=torch.complex64)
    y = torch.randn(n, device='cuda', dtype=torch.complex64)

    # Clone originals
    A_ref = A.clone()
    x_ref = x.clone()
    y_ref = y.clone()

    alpha = complex(1.25, -0.5)
    beta = complex(-0.3, 0.75)

    # cuBLAS parameters
    CUBLAS_FILL_MODE_UPPER = 0
    uplo = CUBLAS_FILL_MODE_UPPER
    lda = n
    incx = 1
    incy = 1

    # Call baseline
    y_out = cublasCsymv_v2(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
    assert y_out is not None

    # PyTorch reference (column-major vs row-major yields A^T, but A is symmetric so equal)
    expected = alpha * (A_ref @ x_ref) + beta * y_ref

    torch.testing.assert_close(y_out, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasCsymv_v2 test passed")