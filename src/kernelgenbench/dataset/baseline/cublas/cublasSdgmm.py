import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_side
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_side

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasSdgmm', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # mode
            ctypes.c_int,  # m
            ctypes.c_void_p,  # n
            ctypes.c_int,  # A
            ctypes.c_void_p,  # lda
            ctypes.c_int,  # x
            ctypes.c_void_p,  # incx
            ctypes.c_int,  # C
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasSdgmm(mode, m, n, A, lda, x, incx, C, ldc):
    '''ctypes cuBLAS C API baseline for cublasSdgmm'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    mode = map_side(mode)
    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Call cuBLAS C API
    func(handle, mode, m, n, A_ptr, lda, x_ptr, incx, C_ptr, ldc)

    return C

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    m_orig, n_orig = 4, 6
    A = torch.randn(m_orig, n_orig, dtype=torch.float32, device='cuda').contiguous()
    A_clone = A.clone()

    CUBLAS_SIDE_LEFT = 0
    CUBLAS_SIDE_RIGHT = 1

    # Column-major trick:
    # Row-major A(m,n) is read as column-major (n,m) by cuBLAS.
    # cuBLAS m_param=n_orig, n_param=m_orig, lda=n_orig, ldc=n_orig

    # --- Right multiplication in row-major: C = A * diag(x_right) ---
    # x_right has n_orig elements (one per column)
    # In column-major view (n,m): RIGHT means scale each column of A_cm by x[col_idx]
    # But columns of A_cm correspond to rows of A_rm.
    # Actually: cuBLAS LEFT on A_cm(n,m) with x(n) scales each row of A_cm => each column of A_rm ✓
    x_right = torch.randn(n_orig, dtype=torch.float32, device='cuda')
    x_right_clone = x_right.clone()
    C_right = torch.empty_like(A)

    out_right = cublasSdgmm(CUBLAS_SIDE_LEFT, n_orig, m_orig, A, n_orig, x_right, 1, C_right, n_orig)
    assert out_right is not None
    expected_right = A_clone * x_right_clone.view(1, -1)
    torch.testing.assert_close(out_right, expected_right, rtol=1e-5, atol=1e-5)

    # --- Left multiplication in row-major: C = diag(x_left) * A ---
    # x_left has m_orig elements (one per row)
    # cuBLAS RIGHT on A_cm(n,m) with x(m): C_cm[i,j] = A_cm[i,j]*x[j]
    # In row-major: C_rm[j,i] = A_rm[j,i]*x[j] => row j of A_rm scaled by x[j] ✓
    x_left = torch.randn(m_orig, dtype=torch.float32, device='cuda')
    x_left_clone = x_left.clone()
    C_left = torch.empty_like(A)

    out_left = cublasSdgmm(CUBLAS_SIDE_RIGHT, n_orig, m_orig, A, n_orig, x_left, 1, C_left, n_orig)
    assert out_left is not None
    expected_left = x_left_clone.view(-1, 1) * A_clone
    torch.testing.assert_close(out_left, expected_left, rtol=1e-5, atol=1e-5)

    print("✓ cublasSdgmm test passed")