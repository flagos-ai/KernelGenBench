import torch
import ctypes

# Load cuBLAS library
libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')

def cublasHgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    """ctypes cuBLAS C API baseline for cublasHgemmStridedBatched"""
    # Create handle
    cublasCreate_v2 = libcublas.cublasCreate_v2
    cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cublasCreate_v2.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    cublasCreate_v2(ctypes.byref(handle))

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    B_ptr = ctypes.c_void_p(B.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # For scalar parameters, create GPU tensor (half precision)
    alpha_gpu = torch.tensor([alpha], dtype=torch.float16, device='cuda')
    beta_gpu = torch.tensor([beta], dtype=torch.float16, device='cuda')
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Set pointer mode to device
    cublasSetPointerMode = libcublas.cublasSetPointerMode_v2
    cublasSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    cublasSetPointerMode.restype = ctypes.c_int
    cublasSetPointerMode(handle, 1)  # CUBLAS_POINTER_MODE_DEVICE

    # Define function signature
    cublasHgemmStridedBatched_func = libcublas.cublasHgemmStridedBatched
    cublasHgemmStridedBatched_func.argtypes = [
        ctypes.c_void_p,            # handle
        ctypes.c_int,               # transa
        ctypes.c_int,               # transb
        ctypes.c_int,               # m
        ctypes.c_int,               # n
        ctypes.c_int,               # k
        ctypes.POINTER(ctypes.c_uint16),  # alpha (device pointer to __half)
        ctypes.POINTER(ctypes.c_uint16),  # A
        ctypes.c_int,               # lda
        ctypes.c_longlong,          # strideA
        ctypes.POINTER(ctypes.c_uint16),  # B
        ctypes.c_int,               # ldb
        ctypes.c_longlong,          # strideB
        ctypes.POINTER(ctypes.c_uint16),  # beta (device pointer to __half)
        ctypes.POINTER(ctypes.c_uint16),  # C
        ctypes.c_int,               # ldc
        ctypes.c_longlong,          # strideC
        ctypes.c_int                # batchCount
    ]
    cublasHgemmStridedBatched_func.restype = ctypes.c_int

    # Call cuBLAS C API
    status = cublasHgemmStridedBatched_func(
        handle, transa, transb, m, n, k,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_uint16)),
        ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_uint16)), lda, ctypes.c_longlong(strideA),
        ctypes.cast(B_ptr, ctypes.POINTER(ctypes.c_uint16)), ldb, ctypes.c_longlong(strideB),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_uint16)),
        ctypes.cast(C_ptr, ctypes.POINTER(ctypes.c_uint16)), ldc, ctypes.c_longlong(strideC),
        batchCount
    )

    # Destroy handle
    cublasDestroy_v2 = libcublas.cublasDestroy_v2
    cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    cublasDestroy_v2.restype = ctypes.c_int
    cublasDestroy_v2(handle)

    return C

if __name__ == "__main__":
    # Test code
    # Problem dimensions
    batchCount = 8
    m, n, k = 32, 32, 32
    alpha, beta = 1.0, 0.0

    # Create test tensors on GPU with correct dtype (float16)
    # Use small integer values to ensure exact representability in float16
    A = torch.randint(-1, 2, (batchCount, m, k), device='cuda').to(torch.float16)
    B = torch.randint(-1, 2, (batchCount, k, n), device='cuda').to(torch.float16)
    C = torch.zeros((batchCount, m, n), dtype=torch.float16, device='cuda')

    # Clone originals for comparison
    A_orig = A.clone()
    B_orig = B.clone()
    C_orig = C.clone()

    # cuBLAS expects column-major; use transposed contiguous tensors
    A_t = A.permute(0, 2, 1).contiguous()  # (batch, k, m)
    B_t = B.permute(0, 2, 1).contiguous()  # (batch, n, k)
    C_t = C.permute(0, 2, 1).contiguous()  # (batch, n, m)

    # Leading dimensions for column-major interpretation
    lda = m
    ldb = k
    ldc = m

    # Strides (in elements) between batches for column-major
    strideA = lda * k  # m * k
    strideB = ldb * n  # k * n
    strideC = ldc * n  # m * n

    # Call baseline function
    result = cublasHgemmStridedBatched(
        0, 0,  # transa, transb = CUBLAS_OP_N
        m, n, k,
        alpha,
        A_t, lda, strideA,
        B_t, ldb, strideB,
        beta,
        C_t, ldc, strideC,
        batchCount
    )

    assert result is not None

    # Compute PyTorch reference (row-major)
    expected = alpha * torch.bmm(A_orig, B_orig) + beta * C_orig

    # Compare: transpose back to row-major for comparison
    torch.testing.assert_close(result.transpose(1, 2), expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasHgemmStridedBatched test passed")