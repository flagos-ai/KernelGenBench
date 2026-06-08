import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, _cuComplex
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, _cuComplex

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasCcopy_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.POINTER(_cuComplex),  # n
            ctypes.c_int,  # x
            ctypes.POINTER(_cuComplex),  # incx
            ctypes.c_int,  # y
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include value so different alpha/beta values get different tensors
    if dtype in (torch.complex64, torch.complex128):
        cache_key = (key, dtype, complex(value))
    else:
        cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasCcopy_v2(n, x, incx, y, incy):
    '''ctypes cuBLAS C API baseline for cublasCcopy_v2'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers and cast to typed pointers
    x_ptr_void = ctypes.c_void_p(x.data_ptr())
    y_ptr_void = ctypes.c_void_p(y.data_ptr())
    x_ptr = ctypes.cast(x_ptr_void, ctypes.POINTER(_cuComplex))
    y_ptr = ctypes.cast(y_ptr_void, ctypes.POINTER(_cuComplex))

    # Call cuBLAS C API
    func(handle, n, x_ptr, incx, y_ptr, incy)

    return y

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    device = 'cuda'

    # Parameters
    n = 5
    incx = 2
    incy = 3

    # Sizes accounting for increments
    size_x = 1 + (n - 1) * incx
    size_y = 1 + (n - 1) * incy

    # Create test tensors on GPU
    x = torch.randn(size_x, dtype=torch.complex64, device=device) + 1j * torch.randn(size_x, dtype=torch.complex64, device=device)
    y = torch.randn(size_y, dtype=torch.complex64, device=device) + 1j * torch.randn(size_y, dtype=torch.complex64, device=device)

    # Clone originals
    x_clone = x.clone()
    y_clone = y.clone()

    # Call baseline
    result = cublasCcopy_v2(n, x, incx, y, incy)
    assert result is not None

    # PyTorch reference
    expected = y_clone.clone()
    expected_slice = expected[::incy]
    x_slice = x_clone[::incx]
    expected_slice[:n] = x_slice[:n]

    # Numerical check
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasCcopy_v2 test passed")