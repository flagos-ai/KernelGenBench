import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func

# Global variables for caching (initialized once, reused)
_cublas_func = None
_cublas_saxpy_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters




def _get_saxpy_func():
    '''Get cublasSaxpy_v2 function with signature set (once)'''
    global _cublas_saxpy_func
    if _cublas_saxpy_func is None:
        _cublas_saxpy_func = get_blas_func("cublasSaxpy_v2")
        _cublas_saxpy_func.argtypes = [
            ctypes.c_void_p, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int
        ]
        _cublas_saxpy_func.restype = ctypes.c_int
    return _cublas_saxpy_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    k = (key, float(value), str(dtype))
    if k not in _scalar_cache:
        _scalar_cache[k] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[k]

def cublasSaxpy_v2(n, alpha, x, incx, y, incy):
    '''ctypes cuBLAS C API baseline for cublasSaxpy_v2: y = alpha * x + y'''
    handle = get_or_create_handle()
    func = _get_saxpy_func()

    # Convert tensors to GPU pointers
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    # Get cached alpha GPU tensor
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    # Call cuBLAS C API
    status = func(
        handle, n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float)), incx,
        ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_float)), incy
    )
    if status != 0:
        raise RuntimeError(f"cublasSaxpy_v2 failed with status {status}")

    return y

if __name__ == "__main__":
    # Test code
    n = 100
    alpha = 2.5
    x = torch.randn(n, dtype=torch.float32, device='cuda')
    y = torch.randn(n, dtype=torch.float32, device='cuda')
    y_original = y.clone()

    result = cublasSaxpy_v2(n, alpha, x, 1, y, 1)

    assert result is not None
    expected = alpha * x + y_original
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasSaxpy_v2 test passed")