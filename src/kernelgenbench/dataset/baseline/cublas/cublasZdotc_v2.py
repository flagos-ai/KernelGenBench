import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, set_pointer_mode, map_op, cuDoubleComplex
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, set_pointer_mode, map_op, cuDoubleComplex

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasZdotc_v2', [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include value so different alpha/beta values get different tensors
    if dtype in (torch.complex64, torch.complex128):
        cache_key = (key, dtype, complex(value))
    else:
        cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasZdotc_v2(n, x, incx, y, incy, result):
    '''ctypes cuBLAS C API baseline for cublasZdotc_v2
    NOTE: cublasZdotc_v2 returns status 15 on driver 470 + cuBLAS 12.4.
    Fall back to cublasZgemm_v2:  result(1×1) = conj(x)^T · y = x^H · y.
    '''
    handle = get_or_create_handle()

    xs = x[::incx][:n].contiguous()
    ys = y[::incy][:n].contiguous()

    set_pointer_mode(handle, 0)  # HOST

    gemm = get_blas_func("cublasZgemm_v2")
    gemm.restype = ctypes.c_int

    alpha = cuDoubleComplex(1.0, 0.0)
    beta  = cuDoubleComplex(0.0, 0.0)

    CUBLAS_OP_C = map_op(2)
    CUBLAS_OP_N = map_op(0)

    status = gemm(
        handle,
        ctypes.c_int(CUBLAS_OP_C), ctypes.c_int(CUBLAS_OP_N),
        ctypes.c_int(1), ctypes.c_int(1), ctypes.c_int(n),
        ctypes.byref(alpha),
        ctypes.c_void_p(xs.data_ptr()), ctypes.c_int(n),
        ctypes.c_void_p(ys.data_ptr()), ctypes.c_int(n),
        ctypes.byref(beta),
        ctypes.c_void_p(result.data_ptr()), ctypes.c_int(1),
    )

    set_pointer_mode(handle, 1)  # DEVICE

    if status != 0:
        raise RuntimeError(f"cublasZdotc_v2 (gemm fallback) failed with status {status}")

    return result

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    size = 30
    incx = 2
    incy = 3
    n = 5

    x = torch.randn(size, device='cuda', dtype=torch.complex128)
    y = torch.randn(size, device='cuda', dtype=torch.complex128)
    # Clone originals for reference
    x_ref = x.clone()
    y_ref = y.clone()

    # Prepare result tensor on GPU
    result = torch.empty(1, device='cuda', dtype=torch.complex128)

    # Call baseline
    out = cublasZdotc_v2(n, x, incx, y, incy, result)
    assert out is not None

    # PyTorch reference computation
    x_slice = x_ref[::incx][:n]
    y_slice = y_ref[::incy][:n]
    expected = (x_slice.conj() * y_slice).sum()

    torch.testing.assert_close(out[0], expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasZdotc_v2 test passed")