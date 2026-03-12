import torch
import ctypes

# Load cuBLAS library
libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')

def cublasDgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    """ctypes cuBLAS C API baseline for cublasDgemmStridedBatched"""
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

    # For scalar parameters, create GPU tensor (double precision)
    alpha_gpu = torch.tensor([alpha], dtype=torch.float64, device='cuda')
    beta_gpu = torch.tensor([beta], dtype=torch.float64, device='cuda')
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Set pointer mode to device
    cublasSetPointerMode = libcublas.cublasSetPointerMode_v2
    cublasSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    cublasSetPointerMode.restype = ctypes.c_int
    cublasSetPointerMode(handle, 1)  # CUBLAS_POINTER_MODE_DEVICE

    # Define function signature
    cublasDgemmStridedBatched_func = libcublas.cublasDgemmStridedBatched
    cublasDgemmStridedBatched_func.argtypes = [
        ctypes.c_void_p,                 # handle
        ctypes.c_int,                    # transa
        ctypes.c_int,                    # transb
        ctypes.c_int,                    # m
        ctypes.c_int,                    # n
        ctypes.c_int,                    # k
        ctypes.POINTER(ctypes.c_double), # alpha
        ctypes.POINTER(ctypes.c_double), # A
        ctypes.c_int,                    # lda
        ctypes.c_longlong,               # strideA
        ctypes.POINTER(ctypes.c_double), # B
        ctypes.c_int,                    # ldb
        ctypes.c_longlong,               # strideB
        ctypes.POINTER(ctypes.c_double), # beta
        ctypes.POINTER(ctypes.c_double), # C
        ctypes.c_int,                    # ldc
        ctypes.c_longlong,               # strideC
        ctypes.c_int                     # batchCount
    ]
    cublasDgemmStridedBatched_func.restype = ctypes.c_int

    # Call cuBLAS C API
    status = cublasDgemmStridedBatched_func(
        handle, transa, transb, m, n, k,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_double)), lda, strideA,
        ctypes.cast(B_ptr, ctypes.POINTER(ctypes.c_double)), ldb, strideB,
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(C_ptr, ctypes.POINTER(ctypes.c_double)), ldc, strideC,
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
    # Dimensions
    batchCount = 8
    m, n, k = 32, 48, 16
    alpha, beta = 1.25, 0.5

    # Create input tensors in row-major on GPU (double precision)
    A_rm = torch.randn(batchCount, m, k, dtype=torch.float64, device='cuda')
    B_rm = torch.randn(batchCount, k, n, dtype=torch.float64, device='cuda')
    C_rm = torch.randn(batchCount, m, n, dtype=torch.float64, device='cuda')
    C_rm_original = C_rm.clone()

    # Convert to column-major by using transposed contiguous tensors
    # cuBLAS expects column-major, so we pass the transpose as contiguous memory
    A_cm = A_rm.transpose(-2, -1).contiguous()  # shape: (batch, k, m) -> column-major (m x k)
    B_cm = B_rm.transpose(-2, -1).contiguous()  # shape: (batch, n, k) -> column-major (k x n)
    C_cm = C_rm.transpose(-2, -1).contiguous()  # shape: (batch, n, m) -> column-major (m x n)

    # Leading dimensions for column-major
    lda = m
    ldb = k
    ldc = m

    # Strides between consecutive matrices in elements
    strideA = k * m
    strideB = n * k
    strideC = m * n

    # cublasOperation_t values: 0 = CUBLAS_OP_N, 1 = CUBLAS_OP_T
    transa = 0
    transb = 0

    # Call baseline
    result_cm = cublasDgemmStridedBatched(
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
    result_rm = result_cm.transpose(-2, -1)

    # PyTorch reference in row-major
    expected = alpha * torch.matmul(A_rm, B_rm) + beta * C_rm_original

    torch.testing.assert_close(result_rm, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasDgemmStridedBatched test passed")