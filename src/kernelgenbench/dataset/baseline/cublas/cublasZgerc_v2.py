import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, cuDoubleComplex
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, cuDoubleComplex

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasZgerc_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # m
            ctypes.POINTER(cuDoubleComplex),  # n
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
    cache_key = (key, dtype, complex(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasZgerc_v2(m, n, alpha, x, incx, y, incy, A, lda):
    '''ctypes cuBLAS C API baseline for cublasZgerc_v2'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    # Convert tensors to GPU pointers
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())
    A_ptr = ctypes.c_void_p(A.data_ptr())

    # Get cached scalar GPU tensor for alpha (complex128)
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.complex128)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    # Call cuBLAS C API
    status = func(
        handle,
        m,
        n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuDoubleComplex)),
        x_ptr,
        incx,
        y_ptr,
        incy,
        A_ptr,
        lda
    )
    if status != 0:
        raise RuntimeError(f"cublasZgerc_v2 failed with status {status}")

    return A

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)

    # Dimensions
    m, n = 3, 4

    # Create test tensors on GPU with correct dtype
    # Use larger buffers to test non-unit strides
    incx = 2
    incy = 3
    x_len = 1 + (m - 1) * incx
    y_len = 1 + (n - 1) * incy

    x_full = torch.randn(x_len, dtype=torch.complex128, device='cuda') + 1j * torch.randn(x_len, dtype=torch.complex128, device='cuda')
    y_full = torch.randn(y_len, dtype=torch.complex128, device='cuda') + 1j * torch.randn(y_len, dtype=torch.complex128, device='cuda')

    # Matrix A
    A = (torch.randn(m, n, dtype=torch.complex128, device='cuda') +
         1j * torch.randn(m, n, dtype=torch.complex128, device='cuda'))

    # Clone originals for comparison
    A_ref = A.clone()
    x_ref = x_full.clone()
    y_ref = y_full.clone()

    # Scalar alpha
    alpha = complex(0.7, -0.3)

    # Call baseline
    result = cublasZgerc_v2(m, n, alpha, x_full, incx, y_full, incy, A, m)
    assert result is not None

    # PyTorch reference considering column-major cuBLAS semantics:
    # Effective vectors with strides
    x_eff = x_ref[::incx].clone()
    y_eff = y_ref[::incy].clone()
    # Compute outer product and map to column-major linear indexing
    update = (alpha * torch.outer(x_eff, y_eff.conj()))  # shape (m, n)
    A_expected = A_ref.clone()
    A_expected.view(-1).add_(update.t().contiguous().view(-1))

    torch.testing.assert_close(result, A_expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasZgerc_v2 test passed")