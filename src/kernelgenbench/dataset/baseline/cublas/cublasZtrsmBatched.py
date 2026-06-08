import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_fill_mode, map_side, map_diag, map_op, cuDoubleComplex
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_fill_mode, map_side, map_diag, map_op, cuDoubleComplex

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasZtrsmBatched', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # side
            ctypes.c_int,  # uplo
            ctypes.c_int,  # trans
            ctypes.c_int,  # diag
            ctypes.c_int,  # m
            ctypes.POINTER(cuDoubleComplex),  # n
            ctypes.POINTER(ctypes.c_void_p),  # alpha (device)
            ctypes.c_int,  # Aarray (device pointer array)
            ctypes.POINTER(ctypes.c_void_p),  # lda
            ctypes.c_int,  # Barray (device pointer array)
            ctypes.c_int,  # ldb
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, complex(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasZtrsmBatched(side, uplo, trans, diag, m, n, alpha, Aarray, lda, Barray, ldb, batchCount):
    '''ctypes cuBLAS C API baseline for cublasZtrsmBatched'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    # Map string enums to ints if necessary
    if isinstance(side, str):
        side = 0 if side == 'L' else 1
    if isinstance(uplo, str):
        uplo = 0 if uplo == 'U' else 1
    if isinstance(trans, str):
        trans = 0 if trans == 'N' else (1 if trans == 'T' else 2)
    if isinstance(diag, str):
        diag = 0 if diag == 'N' else 1
    # Map enums to backend (identity on NVIDIA, remapped on Hygon)
    diag = map_diag(diag)
    side = map_side(side)
    uplo = map_fill_mode(uplo)
    trans = map_op(trans)

    # Device pointer arrays (int64 tensors on GPU containing device pointers)
    Aarray_ptr = ctypes.c_void_p(Aarray.data_ptr())
    Barray_ptr = ctypes.c_void_p(Barray.data_ptr())

    # Alpha scalar on device (complex128)
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.complex128)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(side),
        ctypes.c_int(uplo),
        ctypes.c_int(trans),
        ctypes.c_int(diag),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.cast(Aarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(lda),
        ctypes.cast(Barray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(ldb),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasZtrsmBatched failed with status {status}")
    return Barray

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    device = 'cuda'
    dtype = torch.complex128

    # Parameters
    batchCount = 3
    m = 3
    n = 3
    side = 'L'   # left side
    uplo = 'L'   # A_rm is upper triangular, column-major sees lower triangular
    trans = 'N'  # no transpose
    diag = 'N'   # non-unit diagonal
    alpha = 1.2 - 0.7j

    # Create batched A (triangular) and B matrices and their device pointer arrays
    A_list = []
    B_list = []
    for i in range(batchCount):
        A = torch.randn((m, m), dtype=dtype, device=device)
        # Make A upper triangular and well-conditioned
        A = torch.triu(A)
        A = A + torch.diag(torch.ones(m, dtype=dtype, device=device) * (2.0 + 0.0j))
        B = torch.randn((m, n), dtype=dtype, device=device)
        A_list.append(A)
        B_list.append(B)

    # Clone originals for reference
    A_list_ref = [A.clone() for A in A_list]
    B_list_ref = [B.clone() for B in B_list]

    # Build device pointer arrays (int64 tensors on GPU)
    A_ptrs = torch.tensor([A.data_ptr() for A in A_list], dtype=torch.int64, device=device)
    B_ptrs = torch.tensor([B.data_ptr() for B in B_list], dtype=torch.int64, device=device)

    # Leading dimensions for square case
    lda = m
    ldb = m

    # Call baseline
    result_ptrs = cublasZtrsmBatched(side, uplo, trans, diag, m, n, alpha, A_ptrs, lda, B_ptrs, ldb, batchCount)
    assert result_ptrs is not None

    # PyTorch reference (account for column-major by transposing)
    for i in range(batchCount):
        A_rm = A_list_ref[i]
        B_rm = B_list_ref[i]

        # Column-major views
        A_cm = A_rm.t().contiguous()
        B_cm = B_rm.t().contiguous()

        # Apply op(A)
        if trans == 'N':
            A_eff = A_cm
        elif trans == 'T':
            A_eff = A_cm.transpose(0, 1).contiguous()
        else:  # 'C'
            A_eff = A_cm.transpose(0, 1).conj().contiguous()

        # Solve op(A_cm) * X_cm = alpha * B_cm (left side)
        RHS = alpha * B_cm
        X_cm = torch.linalg.solve(A_eff, RHS)

        # Convert back to row-major result
        expected = X_cm.t().contiguous()

        torch.testing.assert_close(B_list[i], expected, rtol=1e-12, atol=1e-12)

    print("✓ cublasZtrsmBatched test passed")