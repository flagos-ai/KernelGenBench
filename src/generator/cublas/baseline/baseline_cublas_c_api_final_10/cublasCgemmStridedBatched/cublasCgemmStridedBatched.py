import torch
import ctypes

# Load cuBLAS library
libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')

# Define cuComplex structure for complex64
class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

def cublasCgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    """ctypes cuBLAS C API baseline for cublasCgemmStridedBatched"""
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

    # For scalar parameters, create GPU tensor (complex64)
    alpha_gpu = torch.tensor([alpha], dtype=torch.complex64, device='cuda')
    beta_gpu = torch.tensor([beta], dtype=torch.complex64, device='cuda')
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Set pointer mode to device
    cublasSetPointerMode = libcublas.cublasSetPointerMode_v2
    cublasSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    cublasSetPointerMode.restype = ctypes.c_int
    cublasSetPointerMode(handle, 1)  # CUBLAS_POINTER_MODE_DEVICE

    # Define function signature
    cublasCgemmStridedBatched_func = libcublas.cublasCgemmStridedBatched
    cublasCgemmStridedBatched_func.argtypes = [
        ctypes.c_void_p,          # handle
        ctypes.c_int,             # transa
        ctypes.c_int,             # transb
        ctypes.c_int,             # m
        ctypes.c_int,             # n
        ctypes.c_int,             # k
        ctypes.POINTER(cuComplex),# alpha
        ctypes.POINTER(cuComplex),# A
        ctypes.c_int,             # lda
        ctypes.c_longlong,        # strideA
        ctypes.POINTER(cuComplex),# B
        ctypes.c_int,             # ldb
        ctypes.c_longlong,        # strideB
        ctypes.POINTER(cuComplex),# beta
        ctypes.POINTER(cuComplex),# C
        ctypes.c_int,             # ldc
        ctypes.c_longlong,        # strideC
        ctypes.c_int              # batchCount
    ]
    cublasCgemmStridedBatched_func.restype = ctypes.c_int

    # Call cuBLAS C API
    status = cublasCgemmStridedBatched_func(
        handle,
        ctypes.c_int(transa),
        ctypes.c_int(transb),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuComplex)),
        ctypes.cast(A_ptr, ctypes.POINTER(cuComplex)),
        ctypes.c_int(lda),
        ctypes.c_longlong(strideA),
        ctypes.cast(B_ptr, ctypes.POINTER(cuComplex)),
        ctypes.c_int(ldb),
        ctypes.c_longlong(strideB),
        ctypes.cast(beta_ptr, ctypes.POINTER(cuComplex)),
        ctypes.cast(C_ptr, ctypes.POINTER(cuComplex)),
        ctypes.c_int(ldc),
        ctypes.c_longlong(strideC),
        ctypes.c_int(batchCount)
    )
    # Optionally, could check status here

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
    dtype = torch.complex64

    # Dimensions
    batchCount = 5
    m, n, k = 32, 24, 16
    alpha = complex(1.0, 0.0)
    beta = complex(0.5, -0.25)

    # Create random complex inputs in row-major (PyTorch default)
    A_real = torch.randn(batchCount, m, k, device=device, dtype=torch.float32)
    A_imag = torch.randn(batchCount, m, k, device=device, dtype=torch.float32)
    A = (A_real + 1j * A_imag).to(dtype)

    B_real = torch.randn(batchCount, k, n, device=device, dtype=torch.float32)
    B_imag = torch.randn(batchCount, k, n, device=device, dtype=torch.float32)
    B = (B_real + 1j * B_imag).to(dtype)

    C_real = torch.randn(batchCount, m, n, device=device, dtype=torch.float32)
    C_imag = torch.randn(batchCount, m, n, device=device, dtype=torch.float32)
    C = (C_real + 1j * C_imag).to(dtype)
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
    result = cublasCgemmStridedBatched(
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

    torch.testing.assert_close(result_row_major, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasCgemmStridedBatched test passed")