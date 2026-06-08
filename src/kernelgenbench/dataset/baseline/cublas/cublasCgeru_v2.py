import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, cuComplex
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, cuComplex

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasCgeru_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # m
            ctypes.POINTER(cuComplex),  # n
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
    if dtype.is_complex:
        cache_key = (key, dtype, complex(value))
    else:
        cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasCgeru_v2(m, n, alpha, x, incx, y, incy, A, lda):
    '''ctypes cuBLAS C API baseline for cublasCgeru_v2'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())
    A_ptr = ctypes.c_void_p(A.data_ptr())

    # Get cached scalar GPU tensor for complex alpha
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.complex64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    # Call cuBLAS C API
    status = func(
        handle,
        m,
        n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuComplex)),
        x_ptr,
        incx,
        y_ptr,
        incy,
        A_ptr,
        lda
    )
    if status != 0:
        raise RuntimeError(f"cublasCgeru_v2 failed with status {status}")

    return A

if __name__ == "__main__":
    # Test code
    torch.manual_seed(123)

    # Dimensions
    m, n = 4, 3
    incx, incy = 1, 1
    lda = m

    # Create A with column-major memory layout by transposing a contiguous (n, m) tensor
    A_base = torch.randn(n, m, dtype=torch.complex64, device='cuda')
    A = A_base.t()  # shape (m, n), column-major in memory

    # Create vectors x (length m) and y (length n)
    x = torch.randn(m, dtype=torch.complex64, device='cuda')
    y = torch.randn(n, dtype=torch.complex64, device='cuda')

    # Scalar alpha (complex64)
    alpha = complex(0.5, -0.3)

    # Clone originals for reference
    A_before = A.clone()
    x_clone = x.clone()
    y_clone = y.clone()

    # Call baseline
    A_out = cublasCgeru_v2(m, n, alpha, x, incx, y, incy, A, lda)
    assert A_out is not None

    # PyTorch reference (unconjugated rank-1 update): A := alpha * x * y^T + A
    expected = A_before + (alpha * x_clone.unsqueeze(1) * y_clone.unsqueeze(0))

    # Numerical check
    torch.testing.assert_close(A_out, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasCgeru_v2 test passed")