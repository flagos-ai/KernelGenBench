import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_fill_mode, map_side, cuComplex
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_fill_mode, map_side, cuComplex

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasCsymm_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # side
            ctypes.c_int,  # uplo
            ctypes.c_int,  # m
            ctypes.POINTER(cuComplex),  # n
            ctypes.c_void_p,  # alpha
            ctypes.c_int,  # A
            ctypes.c_void_p,  # lda
            ctypes.c_int,  # B
            ctypes.POINTER(cuComplex),  # ldb
            ctypes.c_void_p,  # beta
            ctypes.c_int,  # C
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, complex(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasCsymm_v2(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    '''ctypes cuBLAS C API baseline for cublasCsymm_v2'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    side = map_side(side)
    uplo = map_fill_mode(uplo)
    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    B_ptr = ctypes.c_void_p(B.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensors
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.complex64)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.complex64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Call cuBLAS C API
    status = func(
        handle,
        side,
        uplo,
        m,
        n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuComplex)),
        A_ptr,
        lda,
        B_ptr,
        ldb,
        ctypes.cast(beta_ptr, ctypes.POINTER(cuComplex)),
        C_ptr,
        ldc
    )
    if status != 0:
        raise RuntimeError(f"cublasCsymm_v2 failed with status {status}")

    return C

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    m, n = 3, 4

    # Create symmetric complex64 matrix A (m x m)
    A_rand = torch.randn(m, m, dtype=torch.complex64, device='cuda')
    A_sym = A_rand + A_rand.t()  # symmetric

    B_rm = torch.randn(m, n, dtype=torch.complex64, device='cuda')
    C_rm = torch.randn(m, n, dtype=torch.complex64, device='cuda')

    alpha = 0.7 + 0.3j
    beta = -0.2 + 0.5j

    # Clone originals for reference
    A_ref = A_sym.clone()
    B_ref = B_rm.clone()
    C_ref = C_rm.clone()

    # Column-major trick: transpose all inputs to contiguous column-major layout
    A_cm = A_sym.t().contiguous()  # still symmetric, shape (m, m)
    B_cm = B_rm.t().contiguous()   # shape (n, m)
    C_cm = C_rm.t().contiguous()   # shape (n, m)

    # cuBLAS side=LEFT, uplo=UPPER in column-major
    # With transposed inputs, side=LEFT(cm) becomes side=RIGHT(rm),
    # and uplo=UPPER(cm) becomes uplo=LOWER(rm)
    # cuBLAS sees: C_cm(m×n) = alpha * A_cm(m×m) * B_cm(m×n) + beta * C_cm(m×n)
    # In row-major view: C_rm = (alpha * A * B + beta * C_orig)
    side = 0  # CUBLAS_SIDE_LEFT
    uplo = 1  # CUBLAS_FILL_MODE_UPPER

    lda = m
    ldb = m
    ldc = m

    out = cublasCsymm_v2(side, uplo, m, n, alpha, A_cm, lda, B_cm, ldb, beta, C_cm, ldc)
    assert out is not None

    # cuBLAS result is in C_cm (column-major), transpose back to row-major
    result_rm = out.t().contiguous()

    # PyTorch reference (row-major): C = alpha * A * B + beta * C
    expected = alpha * (A_ref @ B_ref) + beta * C_ref

    torch.testing.assert_close(result_rm, expected, rtol=1e-2, atol=1e-2)
    print("✓ cublasCsymm_v2 test passed")