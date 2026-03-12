import torch
import ctypes

# Load cuBLAS library
libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')

def cublasSgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    """ctypes cuBLAS C API baseline for cublasSgemmStridedBatched"""
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

    # For scalar parameters, create GPU tensor
    alpha_gpu = torch.tensor([alpha], dtype=torch.float32, device='cuda')
    beta_gpu = torch.tensor([beta], dtype=torch.float32, device='cuda')
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Set pointer mode to device
    cublasSetPointerMode = libcublas.cublasSetPointerMode_v2
    cublasSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    cublasSetPointerMode.restype = ctypes.c_int
    cublasSetPointerMode(handle, 1)  # CUBLAS_POINTER_MODE_DEVICE

    # Define function signature
    cublasSgemmStridedBatched_func = libcublas.cublasSgemmStridedBatched
    cublasSgemmStridedBatched_func.argtypes = [
        ctypes.c_void_p,               # handle
        ctypes.c_int,                  # transa
        ctypes.c_int,                  # transb
        ctypes.c_int,                  # m
        ctypes.c_int,                  # n
        ctypes.c_int,                  # k
        ctypes.POINTER(ctypes.c_float),# alpha
        ctypes.POINTER(ctypes.c_float),# A
        ctypes.c_int,                  # lda
        ctypes.c_longlong,             # strideA
        ctypes.POINTER(ctypes.c_float),# B
        ctypes.c_int,                  # ldb
        ctypes.c_longlong,             # strideB
        ctypes.POINTER(ctypes.c_float),# beta
        ctypes.POINTER(ctypes.c_float),# C
        ctypes.c_int,                  # ldc
        ctypes.c_longlong,             # strideC
        ctypes.c_int                   # batchCount
    ]
    cublasSgemmStridedBatched_func.restype = ctypes.c_int

    # Call cuBLAS C API
    status = cublasSgemmStridedBatched_func(
        handle,
        transa, transb,
        m, n, k,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_float)), lda, ctypes.c_longlong(strideA),
        ctypes.cast(B_ptr, ctypes.POINTER(ctypes.c_float)), ldb, ctypes.c_longlong(strideB),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(C_ptr, ctypes.POINTER(ctypes.c_float)), ldc, ctypes.c_longlong(strideC),
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
    torch.manual_seed(0)
    device = 'cuda'
    dtype = torch.float32

    # Dimensions
    batchCount = 5
    m, n, k = 32, 24, 16
    alpha = 1.0
    beta = 0.5

    # Create random inputs in row-major (PyTorch default)
    A = torch.randn(batchCount, m, k, device=device, dtype=dtype)
    B = torch.randn(batchCount, k, n, device=device, dtype=dtype)
    C = torch.randn(batchCount, m, n, device=device, dtype=dtype)
    C_original = C.clone()

    # cuBLAS uses column-major, so transpose to mimic column-major layout
    A_t = A.transpose(-2, -1).contiguous()  # shape (batch, k, m)
    B_t = B.transpose(-2, -1).contiguous()  # shape (batch, n, k)
    C_t = C.transpose(-2, -1).contiguous()  # shape (batch, n, m)

    # Set cuBLAS parameters
    # CUBLAS_OP_N = 0, no transpose because data already arranged for column-major
    transa = 0
    transb = 0

    # Leading dimensions for column-major
    lda = m  # number of rows of A (m x k)
    ldb = k  # number of rows of B (k x n)
    ldc = m  # number of rows of C (m x n)

    # Strides in elements between consecutive matrices in the batch
    strideA = lda * k  # m * k
    strideB = ldb * n  # k * n
    strideC = ldc * n  # m * n

    # Call baseline
    result = cublasSgemmStridedBatched(
        transa, transb, m, n, k,
        alpha,
        A_t, lda, strideA,
        B_t, ldb, strideB,
        beta,
        C_t, ldc, strideC,
        batchCount
    )

    assert result is not None

    # Compare with PyTorch reference in row-major
    expected = alpha * torch.matmul(A, B) + beta * C_original
    # Convert result back to row-major by transposing last two dims
    result_row_major = result.transpose(-2, -1)

    torch.testing.assert_close(result_row_major, expected, rtol=1e-2, atol=1e-2)
    print("✓ cublasSgemmStridedBatched test passed")