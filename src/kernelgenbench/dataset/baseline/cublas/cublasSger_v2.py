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
        _cublas_func = get_blas_func('cublasSger_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # m
            ctypes.POINTER(ctypes.c_float),  # n
            ctypes.c_void_p,  # alpha
            ctypes.c_int,  # x
            ctypes.c_void_p,  # incx
            ctypes.c_int,  # y
            ctypes.c_void_p,  # incy
            ctypes.c_int,  # A
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include value so different alpha/beta values get different tensors
    cache_key = (key, dtype, float(value))  # use complex(value) for complex types
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasSger_v2(m, n, alpha, x, incx, y, incy, A, lda):
    '''ctypes cuBLAS C API baseline for cublasSger_v2'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())
    A_ptr = ctypes.c_void_p(A.data_ptr())

    # Get cached scalar GPU tensor
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    # Call cuBLAS C API
    func(handle, m, n, ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
         x_ptr, incx, y_ptr, incy, A_ptr, lda)

    return A

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    m, n = 5, 7
    incx, incy = 1, 1
    alpha = 0.75

    # Create test data on GPU
    x = torch.randn(m, dtype=torch.float32, device='cuda')
    y = torch.randn(n, dtype=torch.float32, device='cuda')

    # Create A with column-major layout for cuBLAS by using a transposed view
    base = torch.randn(n, m, dtype=torch.float32, device='cuda')  # (n, m) row-major
    A = base.t()  # (m, n) with column-major memory layout (view)

    # Clone originals for reference
    x_ref = x.clone()
    y_ref = y.clone()
    A_ref = A.clone()

    # Compute expected result using PyTorch
    expected = A_ref + alpha * x_ref.unsqueeze(1) * y_ref.unsqueeze(0)

    # Call baseline
    result = cublasSger_v2(m, n, alpha, x, incx, y, incy, A, lda=m)

    # Assertions
    assert result is not None
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasSger_v2 test passed")