import torch
import ctypes

# Load cuBLAS library
libcublas = ctypes.CDLL('/usr/local/cuda/lib64/libcublas.so.12')

def cublasSaxpy_v2(n, alpha, x, incx, y, incy):
    """ctypes cuBLAS C API baseline for cublasSaxpy_v2: y = alpha * x + y"""
    # Create handle
    cublasCreate_v2 = libcublas.cublasCreate_v2
    cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cublasCreate_v2.restype = ctypes.c_int
    handle = ctypes.c_void_p()
    cublasCreate_v2(ctypes.byref(handle))

    # Get tensor pointers
    x_ptr = ctypes.c_void_p(x.data_ptr())
    y_ptr = ctypes.c_void_p(y.data_ptr())

    # Create GPU tensor for scalar alpha
    alpha_gpu = torch.tensor([alpha], dtype=torch.float32, device='cuda')
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())

    # Set pointer mode to device
    cublasSetPointerMode = libcublas.cublasSetPointerMode_v2
    cublasSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
    cublasSetPointerMode.restype = ctypes.c_int
    cublasSetPointerMode(handle, 1)  # CUBLAS_POINTER_MODE_DEVICE

    # Define cublasSaxpy_v2
    cublasSaxpy_v2_func = libcublas.cublasSaxpy_v2
    cublasSaxpy_v2_func.argtypes = [
        ctypes.c_void_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int
    ]
    cublasSaxpy_v2_func.restype = ctypes.c_int

    # Call cuBLAS
    cublasSaxpy_v2_func(
        handle, n,
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(x_ptr, ctypes.POINTER(ctypes.c_float)), incx,
        ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_float)), incy
    )

    # Destroy handle
    cublasDestroy_v2 = libcublas.cublasDestroy_v2
    cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    cublasDestroy_v2.restype = ctypes.c_int
    cublasDestroy_v2(handle)

    return y

if __name__ == "__main__":
    n = 100
    alpha = 2.5
    x = torch.randn(n, dtype=torch.float32, device='cuda')
    y = torch.randn(n, dtype=torch.float32, device='cuda')
    y_original = y.clone()

    result = cublasSaxpy_v2(n, alpha, x, 1, y, 1)

    assert result is not None
    expected = alpha * x + y_original
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasSaxpy_v2 test passed")