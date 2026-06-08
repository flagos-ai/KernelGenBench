import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_fill_mode, map_op, cuComplex, map_data_type
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_fill_mode, map_op, cuComplex, map_data_type

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasCsyrkEx', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # uplo
            ctypes.c_int,  # trans
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # k
            ctypes.c_void_p,  # alpha (cuComplex*)
            ctypes.c_int,  # A (const void*)
            ctypes.c_int,  # Atype (cudaDataType)
            ctypes.POINTER(ctypes.c_float),  # lda
            ctypes.c_void_p,  # beta (cuComplex*)
            ctypes.c_int,  # C (void*)
            ctypes.c_int,  # Ctype (cudaDataType)
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

def cublasCsyrkEx(uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc):
    '''ctypes cuBLAS C API baseline for cublasCsyrkEx'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    uplo = map_fill_mode(uplo)
    trans = map_op(trans)
    Atype = map_data_type(Atype)
    Ctype = map_data_type(Ctype)
    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensors (complex64)
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.complex64)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.complex64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Cast scalar pointers to typed pointers (cuComplex is two floats; typed pointer suffices)
    alpha_typed = ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float))
    beta_typed = ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float))

    # Call cuBLAS C API
    func(handle, uplo, trans, n, k, alpha_typed, A_ptr, Atype, lda, beta_typed, C_ptr, Ctype, ldc)

    return C

if __name__ == "__main__":
    # Constants for enums
    CUBLAS_FILL_MODE_LOWER = 0
    CUBLAS_OP_N = 0
    CUDA_C_32F = 8

    # Test data
    n = 3
    k = 2
    uplo = CUBLAS_FILL_MODE_LOWER
    trans = CUBLAS_OP_N
    alpha = complex(1.0, 0.5)
    beta = complex(0.0, 0.0)

    # Create input matrices (PyTorch row-major)
    A = torch.randn(n, k, dtype=torch.complex64, device='cuda')
    C = torch.zeros(n, n, dtype=torch.complex64, device='cuda')

    # Clone originals
    A_orig = A.clone()
    C_orig = C.clone()

    # Convert to column-major for cuBLAS: row-major (n,k) -> store as (k,n) row-major = (n,k) column-major
    A_cm = A.t().contiguous()
    C_cm = C.t().contiguous()

    # Call baseline with column-major data
    result_cm = cublasCsyrkEx(uplo, trans, n, k, alpha, A_cm, CUDA_C_32F, n, beta, C_cm, CUDA_C_32F, n)
    assert result_cm is not None

    # Convert result back to row-major
    result = result_cm.t().contiguous()

    # PyTorch reference: syrk trans=N computes C = alpha * A * A^T + beta * C
    expected_full = alpha * (A_orig @ A_orig.t()) + beta * C_orig

    # syrk only updates one triangle; uplo=LOWER in column-major = UPPER in row-major
    if uplo == CUBLAS_FILL_MODE_LOWER:
        expected = torch.triu(expected_full)
    else:
        expected = torch.tril(expected_full)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasCsyrkEx test passed")