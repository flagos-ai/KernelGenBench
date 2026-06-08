import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_op
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_op

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters




def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasDgemmStridedBatched', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # transa
            ctypes.c_int,  # transb
            ctypes.c_int,  # m
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_double),  # k
            ctypes.POINTER(ctypes.c_double),  # alpha (device pointer)
            ctypes.c_int,  # A (device pointer)
            ctypes.c_longlong,  # lda
            ctypes.POINTER(ctypes.c_double),  # strideA (in elements)
            ctypes.c_int,  # B (device pointer)
            ctypes.c_longlong,  # ldb
            ctypes.POINTER(ctypes.c_double),  # strideB (in elements)
            ctypes.POINTER(ctypes.c_double),  # beta (device pointer)
            ctypes.c_int,  # C (device pointer)
            ctypes.c_longlong,  # ldc
            ctypes.c_int,  # strideC (in elements)
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasDgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    '''ctypes cuBLAS C API baseline for cublasDgemmStridedBatched'''
    handle = get_or_create_handle()
    func = _get_cublas_func()


    # Map operation enum to backend (cuBLAS: 0/1, hipBLAS: 111/112)
    transa = map_op(transa)
    transb = map_op(transb)
    # Convert string trans to int if needed (N->0, T->1)
    if isinstance(transa, str):
        transa = 0 if transa == 'N' else 1
    if isinstance(transb, str):
        transb = 0 if transb == 'N' else 1

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    B_ptr = ctypes.c_void_p(B.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensor
    alpha_gpu = _get_scalar_gpu('alpha_float64', float(alpha), torch.float64)
    beta_gpu = _get_scalar_gpu('beta_float64', float(beta), torch.float64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(transa),
        ctypes.c_int(transb),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(lda),
        ctypes.c_longlong(strideA),
        ctypes.cast(B_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(ldb),
        ctypes.c_longlong(strideB),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(C_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.c_int(ldc),
        ctypes.c_longlong(strideC),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasDgemmStridedBatched failed with status {status}")

    return C

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    # Dimensions
    batchCount = 4
    m, n, k = 32, 48, 16
    alpha, beta = 1.25, -0.5

    # Create row-major (PyTorch default) tensors
    A = torch.randn(batchCount, m, k, dtype=torch.float64, device='cuda')
    B = torch.randn(batchCount, k, n, dtype=torch.float64, device='cuda')
    C = torch.randn(batchCount, m, n, dtype=torch.float64, device='cuda')

    A_original = A.clone()
    B_original = B.clone()
    C_original = C.clone()

    # Convert to column-major by transposing last two dims and making contiguous
    A_t = A.transpose(-2, -1).contiguous()  # shape: (batch, k, m) -> as col-major (m, k)
    B_t = B.transpose(-2, -1).contiguous()  # shape: (batch, n, k) -> as col-major (k, n)
    C_t = C.transpose(-2, -1).contiguous()  # shape: (batch, n, m) -> as col-major (m, n)

    # cuBLAS op flags
    CUBLAS_OP_N = 0
    transa = CUBLAS_OP_N
    transb = CUBLAS_OP_N

    # Leading dimensions for column-major representations
    lda = m
    ldb = k
    ldc = m

    # Strides (in number of elements) between consecutive matrices in the batch
    strideA = lda * k      # m * k
    strideB = ldb * n      # k * n
    strideC = ldc * n      # m * n

    # Call baseline
    result_t = cublasDgemmStridedBatched(
        transa, transb,
        m, n, k,
        alpha,
        A_t, lda, strideA,
        B_t, ldb, strideB,
        beta,
        C_t, ldc, strideC,
        batchCount
    )

    assert result_t is not None

    # Convert back to row-major for comparison
    result = result_t.transpose(-2, -1).contiguous()

    # PyTorch reference in row-major
    expected = alpha * torch.bmm(A_original, B_original) + beta * C_original

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasDgemmStridedBatched test passed")