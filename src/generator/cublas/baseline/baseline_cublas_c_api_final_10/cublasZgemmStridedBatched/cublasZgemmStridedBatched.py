import torch
import ctypes

# Load cuBLAS library
libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')

# Define cuDoubleComplex type for ctypes
class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

def cublasZgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    """ctypes cuBLAS C API baseline for cublasZgemmStridedBatched"""
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
    alpha_gpu = torch.tensor([alpha], dtype=torch.complex128, device='cuda')
    beta_gpu = torch.tensor([beta], dtype=torch.complex128, device='cuda')
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Set pointer mode to device
    cublasSetPointerMode = libcublas.cublasSetPointerMode_v2
    cublasSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    cublasSetPointerMode.restype = ctypes.c_int
    cublasSetPointerMode(handle, 1)  # CUBLAS_POINTER_MODE_DEVICE

    # Define function signature
    cublasZgemmStridedBatched_func = libcublas.cublasZgemmStridedBatched
    cublasZgemmStridedBatched_func.argtypes = [
        ctypes.c_void_p,            # handle
        ctypes.c_int,               # transa
        ctypes.c_int,               # transb
        ctypes.c_int,               # m
        ctypes.c_int,               # n
        ctypes.c_int,               # k
        ctypes.POINTER(cuDoubleComplex),  # alpha (device pointer)
        ctypes.POINTER(cuDoubleComplex),  # A
        ctypes.c_int,               # lda
        ctypes.c_longlong,          # strideA
        ctypes.POINTER(cuDoubleComplex),  # B
        ctypes.c_int,               # ldb
        ctypes.c_longlong,          # strideB
        ctypes.POINTER(cuDoubleComplex),  # beta (device pointer)
        ctypes.POINTER(cuDoubleComplex),  # C
        ctypes.c_int,               # ldc
        ctypes.c_longlong,          # strideC
        ctypes.c_int                # batchCount
    ]
    cublasZgemmStridedBatched_func.restype = ctypes.c_int

    # Call cuBLAS C API
    status = cublasZgemmStridedBatched_func(
        handle, transa, transb, m, n, k,
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.cast(A_ptr, ctypes.POINTER(cuDoubleComplex)), lda, strideA,
        ctypes.cast(B_ptr, ctypes.POINTER(cuDoubleComplex)), ldb, strideB,
        ctypes.cast(beta_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.cast(C_ptr, ctypes.POINTER(cuDoubleComplex)), ldc, strideC,
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
    dtype = torch.complex128

    # Dimensions
    batchCount = 4
    m, n, k = 32, 16, 20

    # Scalars
    alpha = 1.2 - 0.7j
    beta = -0.3 + 0.5j

    # Create test tensors (row-major in PyTorch)
    A_real = torch.randn(batchCount, m, k, dtype=torch.float64, device=device)
    A_imag = torch.randn(batchCount, m, k, dtype=torch.float64, device=device)
    A = (A_real + 1j * A_imag).to(dtype)

    B_real = torch.randn(batchCount, k, n, dtype=torch.float64, device=device)
    B_imag = torch.randn(batchCount, k, n, dtype=torch.float64, device=device)
    B = (B_real + 1j * B_imag).to(dtype)

    C_real = torch.randn(batchCount, m, n, dtype=torch.float64, device=device)
    C_imag = torch.randn(batchCount, m, n, dtype=torch.float64, device=device)
    C = (C_real + 1j * C_imag).to(dtype)

    C_original = C.clone()

    # Prepare column-major representations by transposing and making contiguous
    # cuBLAS expects column-major; using transposed contiguous arrays with no-transpose ops
    A_t = A.transpose(-2, -1).contiguous()  # shape: (batch, k, m) -> memory matches column-major (m x k)
    B_t = B.transpose(-2, -1).contiguous()  # shape: (batch, n, k) -> memory matches column-major (k x n)
    C_t = C.transpose(-2, -1).contiguous()  # shape: (batch, n, m) -> memory matches column-major (m x n)

    # Leading dimensions for column-major layouts
    lda = m
    ldb = k
    ldc = m

    # Strides (in number of elements, not bytes)
    strideA = m * k
    strideB = k * n
    strideC = m * n

    # Call baseline
    result = cublasZgemmStridedBatched(
        0, 0,  # CUBLAS_OP_N, CUBLAS_OP_N
        m, n, k,
        alpha,
        A_t, lda, strideA,
        B_t, ldb, strideB,
        beta,
        C_t, ldc, strideC,
        batchCount
    )

    assert result is not None

    # Transpose result back to row-major for comparison
    result_rowmajor = result.transpose(-2, -1)

    # PyTorch reference
    expected = alpha * (A @ B) + beta * C_original

    # Numerical check
    torch.testing.assert_close(result_rowmajor, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasZgemmStridedBatched test passed")