import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_op, cuComplex
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_op, cuComplex

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters




def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasCgemvStridedBatched', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # trans
            ctypes.c_int,  # m
            ctypes.POINTER(ctypes.c_float),  # n
            ctypes.POINTER(ctypes.c_float),  # alpha (device pointer to cuComplex)
            ctypes.c_int,  # A (device pointer to cuComplex)
            ctypes.c_longlong,  # lda
            ctypes.POINTER(ctypes.c_float),  # strideA
            ctypes.c_int,  # x (device pointer to cuComplex)
            ctypes.c_longlong,  # incx
            ctypes.POINTER(ctypes.c_float),  # stridex
            ctypes.POINTER(ctypes.c_float),  # beta (device pointer to cuComplex)
            ctypes.c_int,  # y (device pointer to cuComplex)
            ctypes.c_longlong,  # incy
            ctypes.c_int,  # stridey
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, complex(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasCgemvStridedBatched(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    '''ctypes cuBLAS C API baseline for cublasCgemvStridedBatched: batched complex64 GEMV with strided storage'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    # Convert string trans to int if needed (N->0, T->1), then map to backend enum
    if isinstance(trans, str):
        trans = {'N': 0, 'T': 1, 'C': 2}.get(trans.upper(), 0)
    trans_op = map_op(trans)
    
    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())
    
    # Cast to typed pointers (cuComplex is two float32s)
    A_p = ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_float))
    x_p = ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float))
    y_p = ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_float))
    
    # Get cached scalar GPU tensors
    alpha_gpu = _get_scalar_gpu('alpha_complex64', alpha, torch.complex64)
    beta_gpu = _get_scalar_gpu('beta_complex64', beta, torch.complex64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())
    alpha_p = ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float))
    beta_p = ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float))
    
    # Call cuBLAS C API
    status = func(
        handle, trans_op, m, n, alpha_p, A_p, lda, ctypes.c_longlong(strideA),
        x_p, incx, ctypes.c_longlong(stridex), beta_p, y_p, incy, 
        ctypes.c_longlong(stridey), batchCount
    )
    if status != 0:
        raise RuntimeError(f"cublasCgemvStridedBatched failed with status {status}")
    
    return y

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    device = 'cuda'

    # Dimensions
    m, n = 5, 4
    batchCount = 3

    # Create test tensors on GPU with correct dtype
    # Complex A: shape (batch, m, n)
    A_real = torch.randn(batchCount, m, n, device=device)
    A_imag = torch.randn(batchCount, m, n, device=device)
    A = torch.complex(A_real, A_imag).contiguous()

    # x vector: shape (batch, n) for trans='N'
    x_real = torch.randn(batchCount, n, device=device)
    x_imag = torch.randn(batchCount, n, device=device)
    x = torch.complex(x_real, x_imag).contiguous()

    # y vector: shape (batch, m)
    y_real = torch.randn(batchCount, m, device=device)
    y_imag = torch.randn(batchCount, m, device=device)
    y = torch.complex(y_real, y_imag).contiguous()

    # Clone originals for comparison
    A_orig = A.clone()
    x_orig = x.clone()
    y_orig = y.clone()

    # Transpose A to align row-major with cuBLAS column-major expectation
    # We'll pass A_t (n, m) row-major so cuBLAS sees (m, n) column-major.
    A_t = A_orig.transpose(-1, -2).contiguous()

    # Scalars
    alpha = complex(1.0, 0.5)
    beta = complex(0.5, 0.0)

    # cuBLAS parameters
    trans = 'N'
    lda = m
    strideA = m * n
    stridex = n
    stridey = m
    incx = 1
    incy = 1

    # Call baseline
    result = cublasCgemvStridedBatched(
        trans, m, n, alpha, A_t, lda, strideA,
        x_orig, incx, stridex, beta, y, incy, stridey, batchCount
    )

    assert result is not None

    # PyTorch reference: y = alpha * A @ x + beta * y
    expected = alpha * (A_orig @ x_orig.unsqueeze(-1)).squeeze(-1) + beta * y_orig

    torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)
    print("✓ cublasCgemvStridedBatched test passed")
