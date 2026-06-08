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
        _cublas_func = get_blas_func('cublasZswap_v2', [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # Cache key must include the scalar value
    if dtype in (torch.complex64, torch.complex128):
        cache_key = (key, dtype, complex(value))
    else:
        cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasZswap_v2(n, x, incx, y, incy):
    '''ctypes cuBLAS C API baseline for cublasZswap_v2'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    # Call cuBLAS C API
    status = func(handle, n, x_ptr, incx, y_ptr, incy)
    if status != 0:
        raise RuntimeError(f"cublasZswap_v2 failed with status {status}")

    return x

if __name__ == "__main__":
    # Test code
    n = 8
    incx = 1
    incy = 1

    xr = torch.randn(n, device='cuda', dtype=torch.float64)
    xi = torch.randn(n, device='cuda', dtype=torch.float64)
    yr = torch.randn(n, device='cuda', dtype=torch.float64)
    yi = torch.randn(n, device='cuda', dtype=torch.float64)

    x = xr + 1j * xi
    y = yr + 1j * yi

    x0 = x.clone()
    y0 = y.clone()

    result = cublasZswap_v2(n, x, incx, y, incy)
    assert result is not None

    expected_x = y0.clone()
    expected_y = x0.clone()

    torch.testing.assert_close(x, expected_x, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(y, expected_y, rtol=1e-5, atol=1e-5)
    print("✓ cublasZswap_v2 test passed")