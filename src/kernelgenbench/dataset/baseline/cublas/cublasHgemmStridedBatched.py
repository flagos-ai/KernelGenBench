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
        _cublas_func = get_blas_func('cublasHgemmStridedBatched', [
            ctypes.c_void_p,            # handle
            ctypes.c_int,               # transa
            ctypes.c_int,               # transb
            ctypes.c_int,               # m
            ctypes.c_int,               # n
            ctypes.c_int,               # k
            ctypes.POINTER(ctypes.c_uint16),  # alpha (__half*)
            ctypes.POINTER(ctypes.c_uint16),  # A (__half*)
            ctypes.c_int,               # lda
            ctypes.c_longlong,          # strideA
            ctypes.POINTER(ctypes.c_uint16),  # B (__half*)
            ctypes.c_int,               # ldb
            ctypes.c_longlong,          # strideB
            ctypes.POINTER(ctypes.c_uint16),  # beta (__half*)
            ctypes.POINTER(ctypes.c_uint16),  # C (__half*)
            ctypes.c_int,               # ldc
            ctypes.c_longlong,          # strideC
            ctypes.c_int,               # batchCount
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasHgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    '''ctypes cuBLAS C API baseline for cublasHgemmStridedBatched'''
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

    # Get cached scalar GPU tensor (half precision)
    alpha_gpu = _get_scalar_gpu('alpha_h', alpha, torch.float16)
    beta_gpu = _get_scalar_gpu('beta_h', beta, torch.float16)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        int(transa),
        int(transb),
        int(m), int(n), int(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_uint16)),
        ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_uint16)), int(lda), ctypes.c_longlong(int(strideA)),
        ctypes.cast(B_ptr, ctypes.POINTER(ctypes.c_uint16)), int(ldb), ctypes.c_longlong(int(strideB)),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_uint16)),
        ctypes.cast(C_ptr, ctypes.POINTER(ctypes.c_uint16)), int(ldc), ctypes.c_longlong(int(strideC)),
        int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasHgemmStridedBatched failed with status {status}")

    return C

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)

    # GEMM params
    batchCount = 3
    m, n, k = 4, 5, 6
    alpha, beta = 1.0, 0.0

    # Create test tensors (row-major), small integer values to reduce FP16 rounding issues
    A = torch.randint(-2, 3, (batchCount, m, k), device='cuda', dtype=torch.int8).to(torch.float16)
    B = torch.randint(-2, 3, (batchCount, k, n), device='cuda', dtype=torch.int8).to(torch.float16)
    C = torch.zeros((batchCount, m, n), device='cuda', dtype=torch.float16)

    A_orig = A.clone()
    B_orig = B.clone()
    C_orig = C.clone()

    # Prepare column-major storage by using transposed contiguous views
    # cuBLAS expects column-major, passing row-major transpose gives the same memory layout
    A_cm = A.transpose(-2, -1).contiguous()  # shape: (batch, k, m) memory equals column-major (m, k)
    B_cm = B.transpose(-2, -1).contiguous()  # shape: (batch, n, k) memory equals column-major (k, n)
    C_cm = C.transpose(-2, -1).contiguous()  # shape: (batch, n, m) memory equals column-major (m, n)

    # Leading dimensions for column-major views
    lda = m
    ldb = k
    ldc = m

    # Strides in number of elements between consecutive batch matrices (column-major)
    strideA = lda * k        # m * k
    strideB = ldb * n        # k * n
    strideC = ldc * n        # m * n

    # cublasOperation_t: 0 = CUBLAS_OP_N (no transpose)
    transa = 0
    transb = 0

    # Call baseline
    result_cm = cublasHgemmStridedBatched(
        transa, transb,
        m, n, k,
        alpha,
        A_cm, lda, strideA,
        B_cm, ldb, strideB,
        beta,
        C_cm, ldc, strideC,
        batchCount
    )

    assert result_cm is not None

    # Convert result back to row-major for comparison
    result = result_cm.transpose(-2, -1).contiguous()

    # PyTorch reference in row-major
    expected = torch.bmm(A_orig, B_orig) * alpha + C_orig * beta

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasHgemmStridedBatched test passed")