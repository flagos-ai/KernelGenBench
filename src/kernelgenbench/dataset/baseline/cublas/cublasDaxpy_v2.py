import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasDaxpy_v2', [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasDaxpy_v2(n, alpha, x, incx, y, incy):
    '''ctypes cuBLAS C API baseline for cublasDaxpy_v2: y = alpha * x + y'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    # Get cached scalar GPU tensor
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    # Call cuBLAS C API
    func(
        handle, n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_double)), incx,
        ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_double)), incy
    )

    return y

if __name__ == "__main__":
    # Test code
    n = 100
    alpha = 2.5
    x = torch.randn(n, dtype=torch.float64, device='cuda')
    y = torch.randn(n, dtype=torch.float64, device='cuda')
    y_original = y.clone()

    result = cublasDaxpy_v2(n, alpha, x, 1, y, 1)

    assert result is not None
    expected = alpha * x + y_original
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasDaxpy_v2 test passed")