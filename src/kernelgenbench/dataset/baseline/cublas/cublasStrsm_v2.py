import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_op, map_fill_mode, map_side, map_diag
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_op, map_fill_mode, map_side, map_diag

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasStrsm_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # side
            ctypes.c_int,  # uplo
            ctypes.c_int,  # trans
            ctypes.c_int,  # diag
            ctypes.c_int,  # m
            ctypes.POINTER(ctypes.c_float),  # n
            ctypes.c_void_p,  # alpha (device pointer)
            ctypes.c_int,  # A
            ctypes.c_void_p,  # lda
            ctypes.c_int,  # B
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include value so different alpha/beta values get different tensors
    cache_key = (key, dtype, float(value))  # use complex(value) for complex types
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasStrsm_v2(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    '''ctypes cuBLAS C API baseline for cublasStrsm_v2'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    diag = map_diag(diag)
    side = map_side(side)
    uplo = map_fill_mode(uplo)
    trans = map_op(trans)
    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    B_ptr = ctypes.c_void_p(B.data_ptr())

    # Get cached scalar GPU tensor
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    # Call cuBLAS C API
    status = func(
        handle,
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        A_ptr,
        lda,
        B_ptr,
        ldb
    )
    if status != 0:
        raise RuntimeError(f"cublasStrsm_v2 failed with status {status}")

    return B

if __name__ == "__main__":
    # Constants for cuBLAS enums
    CUBLAS_SIDE_LEFT = 0
    CUBLAS_FILL_MODE_LOWER = 0
    CUBLAS_FILL_MODE_UPPER = 1
    CUBLAS_OP_N = 0
    CUBLAS_DIAG_NON_UNIT = 0

    torch.manual_seed(0)

    # Problem sizes
    m = 4
    n = 1  # single RHS to keep column-major and row-major mapping manageable

    # Create upper-triangular A (we will call cuBLAS with uplo=LOWER due to column-major interpretation)
    A_full = torch.randn(m, m, dtype=torch.float32, device='cuda')
    A = torch.triu(A_full)
    A = A + torch.eye(m, dtype=torch.float32, device='cuda') * 3.0  # make well-conditioned

    # Right-hand side B
    B = torch.randn(m, n, dtype=torch.float32, device='cuda')

    # Clone originals
    A_in = A.clone()
    B_in = B.clone()

    alpha = 0.5

    # Call baseline
    result = cublasStrsm_v2(
        CUBLAS_SIDE_LEFT,
        CUBLAS_FILL_MODE_LOWER,  # interpret lower on A_col which corresponds to upper on A row-major
        CUBLAS_OP_N,
        CUBLAS_DIAG_NON_UNIT,
        m,
        n,
        alpha,
        A_in,
        m,     # lda
        B_in,
        m      # ldb
    )

    assert result is not None

    # PyTorch reference (account for column-major interpretation)
    # The result stored in B corresponds to column-major solution X for op(A_col) X = alpha * B_col
    # With A_col = A^T (row-major A transposed), uplo=LOWER means using upper of A (row-major)
    # Compare by transposing: B_out^T should satisfy right-side equation with A (row-major):
    # BT @ A = alpha * B^T
    RHS_T = alpha * B.t()
    # Solve A^T @ Z = RHS_T^T for Z = (B_out^T)^T = B_out
    # But we want BT_expected = (B_out)^T, so compute Z = BT_expected^T then transpose
    Z = torch.linalg.solve_triangular(A.t(), RHS_T.t(), upper=False, left=True, unitriangular=False)
    BT_expected = Z.t()

    torch.testing.assert_close(result.t(), BT_expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasStrsm_v2 test passed")