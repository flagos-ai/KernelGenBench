import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_fill_mode, map_diag, map_op
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_fill_mode, map_diag, map_op

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasStbmv_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # uplo
            ctypes.c_int,  # trans
            ctypes.c_int,  # diag
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # k
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

def cublasStbmv_v2(uplo, trans, diag, n, k, A, lda, x, incx):
    '''ctypes cuBLAS C API baseline for cublasStbmv_v2'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    diag = map_diag(diag)
    uplo = map_fill_mode(uplo)
    trans = map_op(trans)
    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())

    # Call cuBLAS C API
    func(
        handle,
        uplo, trans, diag,
        n, k,
        ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_float)),
        lda,
        ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float)),
        incx
    )

    return x

if __name__ == "__main__":
    # Constants for cuBLAS enums
    CUBLAS_FILL_MODE_LOWER = 0
    CUBLAS_FILL_MODE_UPPER = 1
    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1
    CUBLAS_DIAG_NON_UNIT = 0
    CUBLAS_DIAG_UNIT = 1

    # Test parameters
    n = 6
    k = 2
    lda = k + 1
    uplo = CUBLAS_FILL_MODE_UPPER
    trans = CUBLAS_OP_N
    diag = CUBLAS_DIAG_NON_UNIT
    incx = 1

    # Create a full upper triangular band matrix A_full on GPU
    A_full = torch.zeros((n, n), dtype=torch.float32, device='cuda')
    # Fill only the upper triangular band (including diagonal)
    rand_vals = torch.randn((n, n), dtype=torch.float32, device='cuda')
    for j in range(n):
        i_start = max(0, j - k)
        for i in range(i_start, j + 1):
            A_full[i, j] = rand_vals[i, j]

    # Build band storage AB (lda x n) in column-major semantics
    AB = torch.zeros((lda, n), dtype=torch.float32, device='cuda')
    # For upper triangular band: AB[k + i - j, j] = A_full[i, j]
    for j in range(n):
        i_start = max(0, j - k)
        for i in range(i_start, j + 1):
            AB[k + i - j, j] = A_full[i, j]

    # Create the transposed storage to match column-major expectations
    # Passing AB_t (n x lda) row-major makes cuBLAS see AB (lda x n) column-major
    AB_t = AB.t().contiguous()

    # Vector x
    x = torch.randn(n, dtype=torch.float32, device='cuda')
    x_in = x.clone()

    # Call baseline
    result = cublasStbmv_v2(uplo, trans, diag, n, k, AB_t, lda, x, incx)
    assert result is not None

    # PyTorch reference (column-major adjustment handled by AB_t construction)
    if trans == CUBLAS_OP_N:
        expected = A_full @ x_in
    else:
        expected = A_full.t() @ x_in

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasStbmv_v2 test passed")