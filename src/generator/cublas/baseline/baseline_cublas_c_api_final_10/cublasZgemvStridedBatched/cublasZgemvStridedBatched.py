import torch
import ctypes

# Load cuBLAS library
libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')

# cuDoubleComplex structure for ctypes
class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

def cublasZgemvStridedBatched(trans, m, n, alpha, A, lda, strideA, x, incx, stridex, beta, y, incy, stridey, batchCount):
    """ctypes cuBLAS C API baseline for cublasZgemvStridedBatched: batched complex128 GEMV with strided access"""
    # Create handle
    cublasCreate_v2 = libcublas.cublasCreate_v2
    cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cublasCreate_v2.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    cublasCreate_v2(ctypes.byref(handle))

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    # For scalar parameters, create GPU tensor (complex128)
    alpha_gpu = torch.tensor([alpha], dtype=torch.complex128, device='cuda')
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_gpu = torch.tensor([beta], dtype=torch.complex128, device='cuda')
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Set pointer mode to device
    cublasSetPointerMode = libcublas.cublasSetPointerMode_v2
    cublasSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    cublasSetPointerMode.restype = ctypes.c_int
    cublasSetPointerMode(handle, 1)  # CUBLAS_POINTER_MODE_DEVICE

    # Define function signature
    func = libcublas.cublasZgemvStridedBatched
    func.argtypes = [
        ctypes.c_void_p,                 # handle
        ctypes.c_int,                    # trans
        ctypes.c_int,                    # m
        ctypes.c_int,                    # n
        ctypes.POINTER(cuDoubleComplex), # alpha (device pointer)
        ctypes.POINTER(cuDoubleComplex), # A (device pointer)
        ctypes.c_int,                    # lda
        ctypes.c_longlong,               # strideA
        ctypes.POINTER(cuDoubleComplex), # x (device pointer)
        ctypes.c_int,                    # incx
        ctypes.c_longlong,               # stridex
        ctypes.POINTER(cuDoubleComplex), # beta (device pointer)
        ctypes.POINTER(cuDoubleComplex), # y (device pointer)
        ctypes.c_int,                    # incy
        ctypes.c_longlong,               # stridey
        ctypes.c_int                     # batchCount
    ]
    func.restype = ctypes.c_int

    # Call cuBLAS C API
    status = func(
        handle,
        ctypes.c_int(trans),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.cast(A_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.c_int(lda),
        ctypes.c_longlong(strideA),
        ctypes.cast(x_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.c_int(incx),
        ctypes.c_longlong(stridex),
        ctypes.cast(beta_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.cast(y_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.c_int(incy),
        ctypes.c_longlong(stridey),
        ctypes.c_int(batchCount)
    )

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
    batchCount = 3
    m = 5  # rows in row-major A
    n = 4  # cols in row-major A

    # Create test tensors on GPU with correct dtype
    A = (torch.randn(batchCount, m, n, device=device) + 1j * torch.randn(batchCount, m, n, device=device)).to(torch.complex128).contiguous()
    x = (torch.randn(batchCount, n, device=device) + 1j * torch.randn(batchCount, n, device=device)).to(torch.complex128).contiguous()
    y = (torch.randn(batchCount, m, device=device) + 1j * torch.randn(batchCount, m, device=device)).to(torch.complex128).contiguous()

    # Clone originals
    A_clone = A.clone()
    x_clone = x.clone()
    y_clone = y.clone()

    # Scalars
    alpha = complex(1.3, -0.5)
    beta = complex(-0.2, 0.8)

    # cuBLAS uses column-major; to compute row-major y = A @ x,
    # call with trans = CUBLAS_OP_T and swap m,n in the call.
    CUBLAS_OP_T = 1
    trans = CUBLAS_OP_T

    # In the cuBLAS call:
    # - use m_cublas = n, n_cublas = m
    m_cublas = n
    n_cublas = m

    # Leading dimension (column-major rows of Ac) equals m_cublas = n
    lda = m_cublas

    # Increments and strides (number of elements)
    incx = 1
    incy = 1
    strideA = lda * n_cublas  # equals n * m == m*n, distance between consecutive matrices
    stridex = incx * n        # length of x per batch
    stridey = incy * m        # length of y per batch

    # Flatten batch buffers to single contiguous allocations for strided access
    A_flat = A_clone.reshape(batchCount, -1).contiguous().view(-1)
    x_flat = x_clone.reshape(batchCount, -1).contiguous().view(-1)
    y_flat = y.clone().reshape(batchCount, -1).contiguous().view(-1)  # y is input/output

    # Call baseline
    result = cublasZgemvStridedBatched(
        trans,
        m_cublas,
        n_cublas,
        alpha,
        A_flat,
        lda,
        strideA,
        x_flat,
        incx,
        stridex,
        beta,
        y_flat,
        incy,
        stridey,
        batchCount
    )

    # Reshape result back to (batchCount, m)
    result_view = result.view(batchCount, m)

    # Assert not None
    assert result_view is not None

    # PyTorch reference (row-major): y = alpha * (A @ x) + beta * y
    Ax = torch.einsum('bmn,bn->bm', A_clone, x_clone)
    expected = alpha * Ax + beta * y_clone

    # Numerical check
    torch.testing.assert_close(result_view, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasZgemvStridedBatched test passed")