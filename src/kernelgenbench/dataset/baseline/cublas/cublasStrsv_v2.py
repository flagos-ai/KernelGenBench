import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_op, map_fill_mode, map_diag
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_op, map_fill_mode, map_diag

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasStrsv_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # uplo
            ctypes.c_int,  # trans
            ctypes.c_int,  # diag
            ctypes.POINTER(ctypes.c_float),  # n
            ctypes.c_int,  # A
            ctypes.POINTER(ctypes.c_float),  # lda
            ctypes.c_int,  # x
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasStrsv_v2(uplo, trans, diag, n, A, lda, x, incx):
    '''ctypes cuBLAS C API baseline for cublasStrsv_v2'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    diag = map_diag(diag)
    uplo = map_fill_mode(uplo)
    trans = map_op(trans)
    # Convert tensors to GPU typed pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())
    A_typed = ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_float))
    x_typed = ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float))

    # Call cuBLAS C API
    func(handle, uplo, trans, diag, n, A_typed, lda, x_typed, incx)
    return x

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    n = 4
    # Create an upper-triangular matrix in row-major and make it well-conditioned
    A_full = torch.randn(n, n, dtype=torch.float32, device='cuda')
    A_upper = torch.triu(A_full) + torch.eye(n, dtype=torch.float32, device='cuda') * 5.0

    # Right-hand side vector b
    b = torch.randn(n, dtype=torch.float32, device='cuda')
    x_vec = b.clone()

    # Column-major trick:
    # Row-major A_upper(n,n) is read as column-major A_upper^T by cuBLAS.
    # A_upper^T is lower-triangular. So pass uplo=LOWER, trans=N.
    # cuBLAS solves: A_upper^T * x = b, but with trans=N it solves L*x = b where L = A_upper^T
    # This is equivalent to solving A_upper * x = b (since (A^T)^T = A, and trsv with L,N is same as U,T)
    # Actually: cuBLAS sees L = A_upper^T (lower tri). Solving L*x = b gives x = L^{-1}*b = (A_upper^T)^{-1}*b
    # We want A_upper * x = b => x = A_upper^{-1} * b
    # (A_upper^T)^{-1} != A_upper^{-1} in general. So we need trans=T:
    # cuBLAS solves op(L)*x = b with trans=T => L^T * x = b => A_upper * x = b ✓
    uplo_lower = 0        # CUBLAS_FILL_MODE_LOWER
    trans_T = 1           # CUBLAS_OP_T
    diag_nonunit = 0      # CUBLAS_DIAG_NON_UNIT
    lda = n
    incx = 1

    result = cublasStrsv_v2(uplo_lower, trans_T, diag_nonunit, n, A_upper, lda, x_vec, incx)
    assert result is not None

    # PyTorch reference: solve A_upper x = b
    x_expected = torch.linalg.solve_triangular(A_upper, b.unsqueeze(1), upper=True, left=True, unitriangular=False).squeeze(1)

    torch.testing.assert_close(result, x_expected, rtol=1e-4, atol=1e-4)
    print("✓ cublasStrsv_v2 test passed")