import torch
import ctypes

# Load cuBLAS library
libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')

def cublasDgemvStridedBatched(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    """ctypes cuBLAS C API baseline for cublasDgemvStridedBatched"""
    # Create handle
    cublasCreate_v2 = libcublas.cublasCreate_v2
    cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cublasCreate_v2.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    cublasCreate_v2(ctypes.byref(handle))

    # Prepare device scalar pointers for alpha and beta (double precision)
    alpha_gpu = torch.tensor([alpha], dtype=torch.float64, device='cuda')
    beta_gpu = torch.tensor([beta], dtype=torch.float64, device='cuda')
    alpha_ptr_void = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr_void = ctypes.c_void_p(beta_gpu.data_ptr())
    alpha_ptr = ctypes.cast(alpha_ptr_void, ctypes.POINTER(ctypes.c_double))
    beta_ptr = ctypes.cast(beta_ptr_void, ctypes.POINTER(ctypes.c_double))

    # Convert tensors to GPU pointers
    A_ptr_void = ctypes.c_void_p(A.data_ptr())
    x_ptr_void = ctypes.c_void_p(x.data_ptr())
    y_ptr_void = ctypes.c_void_p(y.data_ptr())
    A_ptr = ctypes.cast(A_ptr_void, ctypes.POINTER(ctypes.c_double))
    x_ptr = ctypes.cast(x_ptr_void, ctypes.POINTER(ctypes.c_double))
    y_ptr = ctypes.cast(y_ptr_void, ctypes.POINTER(ctypes.c_double))

    # Set pointer mode to device
    cublasSetPointerMode = libcublas.cublasSetPointerMode_v2
    cublasSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    cublasSetPointerMode.restype = ctypes.c_int
    cublasSetPointerMode(handle, 1)  # CUBLAS_POINTER_MODE_DEVICE

    # Define function signature
    func = libcublas.cublasDgemvStridedBatched
    func.argtypes = [
        ctypes.c_void_p,              # handle
        ctypes.c_int,                 # trans
        ctypes.c_int,                 # m
        ctypes.c_int,                 # n
        ctypes.POINTER(ctypes.c_double),  # alpha (device pointer)
        ctypes.POINTER(ctypes.c_double),  # A (device pointer)
        ctypes.c_int,                 # lda
        ctypes.c_longlong,            # strideA (in elements)
        ctypes.POINTER(ctypes.c_double),  # x (device pointer)
        ctypes.c_int,                 # incx
        ctypes.c_longlong,            # stridex (in elements)
        ctypes.POINTER(ctypes.c_double),  # beta (device pointer)
        ctypes.POINTER(ctypes.c_double),  # y (device pointer)
        ctypes.c_int,                 # incy
        ctypes.c_longlong,            # stridey (in elements)
        ctypes.c_int                  # batchCount
    ]
    func.restype = ctypes.c_int

    # Call cuBLAS C API
    status = func(handle, trans, m, n,
                  alpha_ptr,
                  A_ptr, lda, ctypes.c_longlong(strideA),
                  x_ptr, incx, ctypes.c_longlong(stridex),
                  beta_ptr,
                  y_ptr, incy, ctypes.c_longlong(stridey),
                  batchCount)

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
    batchCount = 4
    m, n = 5, 3
    incx = 1
    incy = 1
    alpha = 1.25
    beta = -0.5
    trans = 0  # CUBLAS_OP_N

    # Create batched inputs in row-major (PyTorch default)
    A_row = torch.randn(batchCount, m, n, dtype=torch.float64, device=device)
    x = torch.randn(batchCount, n, dtype=torch.float64, device=device)
    y = torch.randn(batchCount, m, dtype=torch.float64, device=device)

    # Clone originals for reference computation
    A_row_ref = A_row.clone()
    x_ref = x.clone()
    y_ref = y.clone()

    # cuBLAS expects column-major. Provide A as A_row.t().contiguous() so that
    # its memory matches the column-major layout of the original A_row.
    A_col = A_row.transpose(-2, -1).contiguous()

    # Compute strides (in elements) between consecutive batches
    strideA = m * n
    stridex = n * incx
    stridey = m * incy

    # Leading dimension (lda) for column-major A is number of rows = m
    lda = m

    # Flatten batched tensors to ensure base pointers reflect contiguous batches
    # (Not strictly necessary as .contiguous() ensures this, but keeps intent clear)
    A_col_flat = A_col
    x_flat = x
    y_flat = y

    # Call baseline
    result = cublasDgemvStridedBatched(
        trans, m, n,
        alpha,
        A_col_flat, lda, strideA,
        x_flat, incx, stridex,
        beta,
        y_flat, incy, stridey,
        batchCount
    )

    # Assert not None
    assert result is not None

    # PyTorch reference (row-major)
    expected = alpha * torch.bmm(A_row_ref, x_ref.unsqueeze(-1)).squeeze(-1) + beta * y_ref

    # Numerical check
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasDgemvStridedBatched test passed")