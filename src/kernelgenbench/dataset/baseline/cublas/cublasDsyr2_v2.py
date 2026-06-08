import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_fill_mode
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_fill_mode

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasDsyr2_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # uplo (cublasFillMode_t)
            ctypes.POINTER(ctypes.c_double),  # n
            ctypes.c_void_p,  # alpha (device pointer)
            ctypes.c_int,  # x (device pointer)
            ctypes.c_void_p,  # incx
            ctypes.c_int,  # y (device pointer)
            ctypes.c_void_p,  # incy
            ctypes.c_int,  # A (device pointer)
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasDsyr2_v2(uplo, n, alpha, x, incx, y, incy, A, lda):
    '''ctypes cuBLAS C API baseline for cublasDsyr2_v2'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    uplo = map_fill_mode(uplo)
    # Convert tensors to GPU pointers
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())
    A_ptr = ctypes.c_void_p(A.data_ptr())

    # Get cached scalar GPU tensor
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    # Call cuBLAS C API
    func(
        handle,
        uplo,
        n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_double)),
        x_ptr,
        incx,
        y_ptr,
        incy,
        A_ptr,
        lda
    )

    return A

if __name__ == "__main__":
    # Test code
    torch.manual_seed(1234)
    n = 5
    uplo = 1  # 1 = upper, 0 = lower
    alpha = 0.75
    incx = 1
    incy = 1
    lda = n

    # Create test data on GPU
    A = torch.randn(n, n, dtype=torch.float64, device='cuda')
    x = torch.randn(n, dtype=torch.float64, device='cuda')
    y = torch.randn(n, dtype=torch.float64, device='cuda')

    # Clone originals
    A0 = A.clone()
    x0 = x.clone()
    y0 = y.clone()

    # Call baseline
    result = cublasDsyr2_v2(uplo, n, alpha, x, incx, y, incy, A, lda)
    assert result is not None

    # PyTorch reference considering column-major semantics:
    # cuBLAS interprets A as column-major, which corresponds to A0.t() in PyTorch row-major memory.
    A_col0 = A0.t().contiguous()
    x_col = x0
    y_col = y0
    M_col = alpha * (x_col.view(n, 1) @ y_col.view(1, n) + y_col.view(n, 1) @ x_col.view(1, n))

    if uplo == 1:
        A_col_expected = torch.triu(A_col0 + M_col) + torch.tril(A_col0, -1)
    else:
        A_col_expected = torch.tril(A_col0 + M_col) + torch.triu(A_col0, 1)

    expected = A_col_expected.t().contiguous()

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasDsyr2_v2 test passed")