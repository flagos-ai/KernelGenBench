import torch
import ctypes

# Load cuBLAS library
libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')

def cublasSgemvStridedBatched(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    """ctypes cuBLAS C API baseline for cublasSgemvStridedBatched"""
    # Basic checks
    if not (A.is_cuda and x.is_cuda and y.is_cuda):
        raise ValueError("A, x, y must be CUDA tensors")
    if A.dtype != torch.float32 or x.dtype != torch.float32 or y.dtype != torch.float32:
        raise ValueError("A, x, y must be float32 tensors")

    # Create handle
    cublasCreate_v2 = libcublas.cublasCreate_v2
    cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cublasCreate_v2.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    status = cublasCreate_v2(ctypes.byref(handle))
    if status != 0:
        raise RuntimeError(f"cublasCreate_v2 failed with status {status}")

    try:
        # For scalar parameters, create GPU tensor and get pointer
        alpha_gpu = torch.tensor([alpha], dtype=torch.float32, device='cuda')
        beta_gpu = torch.tensor([beta], dtype=torch.float32, device='cuda')
        alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
        beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

        # Convert tensors to GPU pointers
        A_ptr = ctypes.c_void_p(A.data_ptr())
        x_ptr = ctypes.c_void_p(x.data_ptr())
        y_ptr = ctypes.c_void_p(y.data_ptr())

        # Set pointer mode to device
        cublasSetPointerMode = libcublas.cublasSetPointerMode_v2
        cublasSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        cublasSetPointerMode.restype = ctypes.c_int
        status = cublasSetPointerMode(handle, 1)  # CUBLAS_POINTER_MODE_DEVICE
        if status != 0:
            raise RuntimeError(f"cublasSetPointerMode_v2 failed with status {status}")

        # Define function signature
        func = libcublas.cublasSgemvStridedBatched
        func.argtypes = [
            ctypes.c_void_p,                # handle
            ctypes.c_int,                   # trans
            ctypes.c_int,                   # m
            ctypes.c_int,                   # n
            ctypes.POINTER(ctypes.c_float), # alpha
            ctypes.POINTER(ctypes.c_float), # A
            ctypes.c_int,                   # lda
            ctypes.c_longlong,              # strideA
            ctypes.POINTER(ctypes.c_float), # x
            ctypes.c_int,                   # incx
            ctypes.c_longlong,              # stridex
            ctypes.POINTER(ctypes.c_float), # beta
            ctypes.POINTER(ctypes.c_float), # y
            ctypes.c_int,                   # incy
            ctypes.c_longlong,              # stridey
            ctypes.c_int                    # batchCount
        ]
        func.restype = ctypes.c_int

        # Cast pointers to typed pointers
        alpha_ptr_cast = ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float))
        beta_ptr_cast = ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float))
        A_ptr_cast = ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_float))
        x_ptr_cast = ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float))
        y_ptr_cast = ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_float))

        # Call cuBLAS C API
        status = func(
            handle,
            int(trans),
            int(m),
            int(n),
            alpha_ptr_cast,
            A_ptr_cast,
            int(lda),
            ctypes.c_longlong(int(strideA)),
            x_ptr_cast,
            int(incx),
            ctypes.c_longlong(int(stridex)),
            beta_ptr_cast,
            y_ptr_cast,
            int(incy),
            ctypes.c_longlong(int(stridey)),
            int(batchCount)
        )
        if status != 0:
            raise RuntimeError(f"cublasSgemvStridedBatched failed with status {status}")
    finally:
        # Destroy handle
        cublasDestroy_v2 = libcublas.cublasDestroy_v2
        cublasDestroy_v2.argtypes = [ctypes.c_void_p]
        cublasDestroy_v2.restype = ctypes.c_int
        cublasDestroy_v2(handle)

    return y

if __name__ == "__main__":
    # Constants for cublasOperation_t
    CUBLAS_OP_N = 0
    CUBLAS_OP_T = 1

    torch.manual_seed(0)

    # Test data
    device = 'cuda'
    dtype = torch.float32
    batchCount = 4
    m = 5
    n = 7

    # Create batched tensors (row-major in PyTorch)
    A = torch.randn(batchCount, m, n, device=device, dtype=dtype).contiguous()
    x = torch.randn(batchCount, n, device=device, dtype=dtype).contiguous()
    y = torch.randn(batchCount, m, device=device, dtype=dtype).contiguous()

    # Clone originals
    A0 = A.clone()
    x0 = x.clone()
    y0 = y.clone()

    alpha = 1.25
    beta = -0.5

    # We want to compute y = alpha * A @ x + beta * y in row-major (PyTorch) for each batch.
    # cuBLAS expects column-major. Using the memory equivalence, pass A as if it were B = A^T in column-major:
    # Call cuBLAS with trans' = OP_T, m' = n, n' = m, lda' = n.
    trans = CUBLAS_OP_T
    m_cu = n
    n_cu = m
    lda = n
    incx = 1
    incy = 1

    # Strides in elements between batches
    strideA = m * n
    stridex = n
    stridey = m

    # Call baseline
    result = cublasSgemvStridedBatched(
        trans, m_cu, n_cu, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount
    )

    assert result is not None

    # PyTorch reference: per-batch y = alpha * (A @ x) + beta * y0
    expected = alpha * torch.bmm(A0, x0.unsqueeze(2)).squeeze(2) + beta * y0

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasSgemvStridedBatched test passed")