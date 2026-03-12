import torch
import ctypes

# Load cuBLAS library
libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')

def cublasCgemvStridedBatched(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    """ctypes cuBLAS C API baseline for cublasCgemvStridedBatched: batched complex64 GEMV with strided storage"""
    # Create handle
    cublasCreate_v2 = libcublas.cublasCreate_v2
    cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cublasCreate_v2.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    cublasCreate_v2(ctypes.byref(handle))

    # Map trans to cublasOperation_t
    def _to_cublas_op(t):
        if isinstance(t, int):
            return t
        t = str(t).upper()
        if t == 'N':
            return 0
        elif t == 'T':
            return 1
        elif t == 'C':
            return 2
        else:
            raise ValueError("Invalid trans value. Use 'N', 'T', or 'C'.")

    trans_op = _to_cublas_op(trans)

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    # Cast to typed pointers (cuComplex is two float32s, so use POINTER(c_float))
    A_p = ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_float))
    x_p = ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float))
    y_p = ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_float))

    # For scalar parameters, create GPU tensor (complex64) and extract pointer
    alpha_gpu = torch.tensor([alpha], dtype=torch.complex64, device='cuda')
    beta_gpu = torch.tensor([beta], dtype=torch.complex64, device='cuda')
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())
    alpha_p = ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float))
    beta_p = ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float))

    # Set pointer mode to device
    cublasSetPointerMode = libcublas.cublasSetPointerMode_v2
    cublasSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    cublasSetPointerMode.restype = ctypes.c_int
    cublasSetPointerMode(handle, 1)  # CUBLAS_POINTER_MODE_DEVICE

    # Define function signature
    func = libcublas.cublasCgemvStridedBatched
    func.argtypes = [
        ctypes.c_void_p,               # handle
        ctypes.c_int,                  # trans
        ctypes.c_int,                  # m
        ctypes.c_int,                  # n
        ctypes.POINTER(ctypes.c_float),# alpha (device pointer to cuComplex)
        ctypes.POINTER(ctypes.c_float),# A (device pointer to cuComplex)
        ctypes.c_int,                  # lda
        ctypes.c_longlong,             # strideA
        ctypes.POINTER(ctypes.c_float),# x (device pointer to cuComplex)
        ctypes.c_int,                  # incx
        ctypes.c_longlong,             # stridex
        ctypes.POINTER(ctypes.c_float),# beta (device pointer to cuComplex)
        ctypes.POINTER(ctypes.c_float),# y (device pointer to cuComplex)
        ctypes.c_int,                  # incy
        ctypes.c_longlong,             # stridey
        ctypes.c_int                   # batchCount
    ]
    func.restype = ctypes.c_int

    # Call cuBLAS C API
    _status = func(handle, trans_op, m, n, alpha_p, A_p, lda, ctypes.c_longlong(strideA),
                   x_p, incx, ctypes.c_longlong(stridex), beta_p, y_p, incy, ctypes.c_longlong(stridey), batchCount)

    # Destroy handle
    cublasDestroy_v2 = libcublas.cublasDestroy_v2
    cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    cublasDestroy_v2.restype = ctypes.c_int
    cublasDestroy_v2(handle)

    return y

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    device = 'cuda'

    # Dimensions
    m, n = 5, 4
    batchCount = 3

    # Create test tensors on GPU with correct dtype
    # Complex A: shape (batch, m, n)
    A_real = torch.randn(batchCount, m, n, device=device)
    A_imag = torch.randn(batchCount, m, n, device=device)
    A = torch.complex(A_real, A_imag).contiguous()

    # x vector: shape (batch, n) for trans='N'
    x_real = torch.randn(batchCount, n, device=device)
    x_imag = torch.randn(batchCount, n, device=device)
    x = torch.complex(x_real, x_imag).contiguous()

    # y vector: shape (batch, m)
    y_real = torch.randn(batchCount, m, device=device)
    y_imag = torch.randn(batchCount, m, device=device)
    y = torch.complex(y_real, y_imag).contiguous()

    # Clone originals for comparison
    A_orig = A.clone()
    x_orig = x.clone()
    y_orig = y.clone()

    # Transpose A to align row-major with cuBLAS column-major expectation
    # We'll pass A_t (n, m) row-major so cuBLAS sees (m, n) column-major.
    A_t = A_orig.transpose(-1, -2).contiguous()

    # Parameters
    trans = 'N'
    alpha = complex(1.1, -0.3)
    beta = complex(-0.2, 0.7)

    # Leading dimension for column-major A is m
    lda = m

    # Strides (in number of elements, not bytes)
    strideA = lda * n  # m * n elements per batch of A
    incx = 1
    incy = 1
    stridex = n * incx  # length of x per batch
    stridey = m * incy  # length of y per batch

    # Call baseline
    y_out = cublasCgemvStridedBatched(trans, m, n, alpha, A_t, lda, strideA, x_orig, incx, stridex, beta, y.clone(), incy, stridey, batchCount)
    assert y_out is not None

    # PyTorch reference computation (accounting for column-major by using original A)
    # y = alpha * A @ x + beta * y, per batch
    expected = alpha * (A_orig @ x_orig.unsqueeze(-1)).squeeze(-1) + beta * y_orig

    torch.testing.assert_close(y_out, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasCgemvStridedBatched test passed")