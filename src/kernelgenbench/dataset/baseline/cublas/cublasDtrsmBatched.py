import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_fill_mode, map_side, map_diag, map_op
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_fill_mode, map_side, map_diag, map_op

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasDtrsmBatched', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # side
            ctypes.c_int,  # uplo
            ctypes.c_int,  # trans
            ctypes.c_int,  # diag
            ctypes.c_int,  # m
            ctypes.POINTER(ctypes.c_double),  # n
            ctypes.POINTER(ctypes.c_void_p),  # alpha (device)
            ctypes.c_int,  # Aarray (device pointer array)
            ctypes.POINTER(ctypes.c_void_p),  # lda
            ctypes.c_int,  # Barray (device pointer array)
            ctypes.c_int,  # ldb
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasDtrsmBatched(side, uplo, trans, diag, m, n, alpha, Aarray, lda, Barray, ldb, batchCount):
    '''ctypes cuBLAS C API baseline for cublasDtrsmBatched'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    diag = map_diag(diag)
    side = map_side(side)
    uplo = map_fill_mode(uplo)
    trans = map_op(trans)
    # Aarray/Barray are int64 tensors on GPU holding device pointers
    Aarray_ptr = ctypes.c_void_p(Aarray.data_ptr())
    Barray_ptr = ctypes.c_void_p(Barray.data_ptr())

    # Get cached scalar GPU tensor for alpha
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(side),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(Aarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(lda),
        ctypes.cast(Barray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(ldb),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasDtrsmBatched failed with status {status}")

    return Barray

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    dtype = torch.float64
    device = 'cuda'

    batchCount = 2
    m = 4
    n = 3
    alpha = 0.7

    CUBLAS_SIDE_LEFT = 0
    CUBLAS_FILL_MODE_UPPER = 1
    CUBLAS_OP_N = 0
    CUBLAS_DIAG_NON_UNIT = 0

    side = CUBLAS_SIDE_LEFT
    uplo = CUBLAS_FILL_MODE_UPPER
    trans = CUBLAS_OP_N
    diag = CUBLAS_DIAG_NON_UNIT

    As = []
    Bs_cm = []
    Bs_orig = []
    for i in range(batchCount):
        diag_vals = 1.0 + torch.rand(m, dtype=dtype, device=device)
        A = torch.diag(diag_vals).to(device=device, dtype=dtype)
        B_rm = torch.randn(m, n, dtype=dtype, device=device)

        As.append(A)
        Bs_orig.append(B_rm.clone())
        # Convert to column-major: .t().contiguous() gives (n,m) contiguous
        # cuBLAS reads it as (m,n) column-major with ldb=m
        Bs_cm.append(B_rm.t().contiguous())

    Aarray = torch.tensor([A.data_ptr() for A in As], dtype=torch.int64, device=device)
    Barray = torch.tensor([B.data_ptr() for B in Bs_cm], dtype=torch.int64, device=device)

    lda = m
    ldb = m

    result = cublasDtrsmBatched(side, uplo, trans, diag, m, n, alpha, Aarray, lda, Barray, ldb, batchCount)
    assert result is not None

    # side=LEFT, uplo=UPPER, trans=N: solves upper(A)*X = alpha*B
    # A is diagonal (upper part = diagonal), so X = alpha * inv(A) * B
    for i in range(batchCount):
        inv_diag = 1.0 / torch.diag(As[i])
        # Result is in Bs_cm[i] (column-major), transpose back to row-major
        result_rm = Bs_cm[i].t().contiguous()
        expected = alpha * (inv_diag.view(-1, 1) * Bs_orig[i])
        torch.testing.assert_close(result_rm, expected, rtol=1e-5, atol=1e-5)

    print("✓ cublasDtrsmBatched test passed")