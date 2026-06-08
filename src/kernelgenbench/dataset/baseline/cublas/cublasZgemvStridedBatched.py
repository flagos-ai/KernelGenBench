import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_op, cuDoubleComplex
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_op, cuDoubleComplex

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters




def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasZgemvStridedBatched', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # trans
            ctypes.c_int,  # m
            ctypes.POINTER(ctypes.c_double),  # n
            ctypes.POINTER(ctypes.c_double),  # alpha (device pointer to cuDoubleComplex)
            ctypes.c_int,  # A (device pointer to cuDoubleComplex)
            ctypes.c_longlong,  # lda
            ctypes.POINTER(ctypes.c_double),  # strideA
            ctypes.c_int,  # x (device pointer to cuDoubleComplex)
            ctypes.c_longlong,  # incx
            ctypes.POINTER(ctypes.c_double),  # stridex
            ctypes.POINTER(ctypes.c_double),  # beta (device pointer to cuDoubleComplex)
            ctypes.c_int,  # y (device pointer to cuDoubleComplex)
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

def cublasZgemvStridedBatched(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    '''ctypes cuBLAS C API baseline for cublasZgemvStridedBatched: batched complex128 GEMV with strided storage'''
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
    
    # Cast to typed pointers (cuDoubleComplex is two float64s)
    A_p = ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_double))
    x_p = ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_double))
    y_p = ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_double))
    
    # Get cached scalar GPU tensors
    alpha_gpu = _get_scalar_gpu('alpha_complex128', alpha, torch.complex128)
    beta_gpu = _get_scalar_gpu('beta_complex128', beta, torch.complex128)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())
    alpha_p = ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_double))
    beta_p = ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_double))
    
    # Call cuBLAS C API
    status = func(
        handle, trans_op, m, n, alpha_p, A_p, lda, ctypes.c_longlong(strideA),
        x_p, incx, ctypes.c_longlong(stridex), beta_p, y_p, incy, 
        ctypes.c_longlong(stridey), batchCount
    )
    if status != 0:
        raise RuntimeError(f"cublasZgemvStridedBatched failed with status {status}")
    
    return y

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    device = 'cuda'

    # Dimensions
    m, n = 5, 4
    batchCount = 3

    # Create test tensors on GPU with correct dtype (complex128)
    A_real = torch.randn(batchCount, m, n, device=device, dtype=torch.float64)
    A_imag = torch.randn(batchCount, m, n, device=device, dtype=torch.float64)
    A = torch.complex(A_real, A_imag).contiguous()

    x_real = torch.randn(batchCount, n, device=device, dtype=torch.float64)
    x_imag = torch.randn(batchCount, n, device=device, dtype=torch.float64)
    x = torch.complex(x_real, x_imag).contiguous()

    y_real = torch.randn(batchCount, m, device=device, dtype=torch.float64)
    y_imag = torch.randn(batchCount, m, device=device, dtype=torch.float64)
    y = torch.complex(y_real, y_imag).contiguous()

    # Clone originals for comparison
    A_orig = A.clone()
    x_orig = x.clone()
    y_orig = y.clone()

    # Transpose A to align row-major with cuBLAS column-major expectation
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
    result = cublasZgemvStridedBatched(
        trans, m, n, alpha, A_t, lda, strideA,
        x_orig, incx, stridex, beta, y, incy, stridey, batchCount
    )

    assert result is not None

    # PyTorch reference: y = alpha * A @ x + beta * y
    expected = alpha * (A_orig @ x_orig.unsqueeze(-1)).squeeze(-1) + beta * y_orig

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasZgemvStridedBatched test passed")
