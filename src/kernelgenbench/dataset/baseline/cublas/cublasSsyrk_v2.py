import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_fill_mode, map_op
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_fill_mode, map_op

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasSsyrk_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # uplo
            ctypes.c_int,  # trans
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # k
            ctypes.c_void_p,  # alpha
            ctypes.c_int,  # A
            ctypes.POINTER(ctypes.c_float),  # lda
            ctypes.c_void_p,  # beta
            ctypes.c_int,  # C
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include value so different alpha/beta values get different tensors
    cache_key = (key, dtype, float(value))  # use complex(value) for complex types
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasSsyrk_v2(uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    '''ctypes cuBLAS C API baseline for cublasSsyrk_v2'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    uplo = map_fill_mode(uplo)
    trans = map_op(trans)
    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensors
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Call cuBLAS C API
    status = func(
        handle,
        uplo,
        trans,
        n,
        k,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        A_ptr,
        lda,
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float)),
        C_ptr,
        ldc
    )
    if status != 0:
        raise RuntimeError(f"cublasSsyrk_v2 failed with status {status}")

    return C

if __name__ == "__main__":
    # Constants for cuBLAS enums
    CUBLAS_FILL_MODE_LOWER = 0
    CUBLAS_FILL_MODE_UPPER = 1
    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1
    CUBLAS_OP_C = 2

    # Test data
    torch.manual_seed(0)
    n = 5
    k = 3
    alpha = 0.75
    beta = 0.25

    A = torch.randn(n, k, dtype=torch.float32, device='cuda').contiguous()
    C = torch.randn(n, n, dtype=torch.float32, device='cuda').contiguous()
    C_orig = C.clone()

    # Parameters adjusted for column-major expectations:
    # Using trans = T so that cuBLAS computes A^T * A with A treated as column-major (k x n)
    uplo = CUBLAS_FILL_MODE_LOWER  # cuBLAS lower triangle corresponds to upper triangle in row-major
    trans = CUBLAS_OP_T
    lda = k  # leading dimension when trans==T (rows of A in column-major)
    ldc = n  # leading dimension for C (rows in column-major)

    # Call baseline
    result = cublasSsyrk_v2(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)
    assert result is not None

    # PyTorch reference (row-major): update upper triangular of C with alpha*A@A.T + beta*C
    S_full = alpha * (A @ A.t()) + beta * C_orig
    expected = C_orig.clone()
    upper_mask = torch.ones(n, n, dtype=torch.bool, device='cuda').triu()
    expected[upper_mask] = S_full[upper_mask]

    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
    print("✓ cublasSsyrk_v2 test passed")