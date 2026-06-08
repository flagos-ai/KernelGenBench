import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, set_pointer_mode, map_op
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, set_pointer_mode, map_op

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasSdot_v2', [
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
    cache_key = (key, dtype, float(value))  # use complex(value) for complex types
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasSdot_v2(n, x, incx, y, incy, result):
    '''ctypes cuBLAS C API baseline for cublasSdot_v2
    NOTE: cublasSdot_v2 returns CUBLAS_STATUS_NOT_SUPPORTED (15) on
    driver 470 + cuBLAS 12.4.  Fall back to cublasSgemm_v2 which
    computes the same dot product as  C(1×1) = x^T · y.
    For non-unit strides the strided elements are first gathered into
    contiguous buffers so that lda/ldb >= k.
    '''
    handle = get_or_create_handle()

    # --- gather contiguous vectors --------------------------------
    xs = x[::incx][:n].contiguous()
    ys = y[::incy][:n].contiguous()

    # --- switch to HOST pointer mode for gemm alpha/beta ----------
    set_pointer_mode(handle, 0)  # HOST

    # --- cublasSgemm_v2: C(1×1) = alpha * A^T(1×n) * B(n×1) -----
    gemm = get_blas_func("cublasSgemm_v2")
    gemm.restype = ctypes.c_int

    alpha = ctypes.c_float(1.0)
    beta  = ctypes.c_float(0.0)

    CUBLAS_OP_T = map_op(1)
    CUBLAS_OP_N = map_op(0)

    status = gemm(
        handle,
        ctypes.c_int(CUBLAS_OP_T), ctypes.c_int(CUBLAS_OP_N),
        ctypes.c_int(1), ctypes.c_int(1), ctypes.c_int(n),
        ctypes.byref(alpha),
        ctypes.c_void_p(xs.data_ptr()), ctypes.c_int(n),
        ctypes.c_void_p(ys.data_ptr()), ctypes.c_int(n),
        ctypes.byref(beta),
        ctypes.c_void_p(result.data_ptr()), ctypes.c_int(1),
    )

    # --- restore DEVICE pointer mode ------------------------------
    set_pointer_mode(handle, 1)  # DEVICE

    if status != 0:
        raise RuntimeError(f"cublasSdot_v2 (gemm fallback) failed with status {status}")

    return result

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    n = 16
    incx = 2
    incy = 3
    len_x = 1 + (n - 1) * abs(incx)
    len_y = 1 + (n - 1) * abs(incy)

    x = torch.randn(len_x, dtype=torch.float32, device='cuda')
    y = torch.randn(len_y, dtype=torch.float32, device='cuda')
    result = torch.zeros(1, dtype=torch.float32, device='cuda')

    # Clone originals
    x_clone = x.clone()
    y_clone = y.clone()

    # Call baseline
    out = cublasSdot_v2(n, x, incx, y, incy, result)
    assert out is not None

    # PyTorch reference with strides
    xs = x_clone[::incx][:n]
    ys = y_clone[::incy][:n]
    expected = (xs * ys).sum().reshape(1)

    # Numerical check
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasSdot_v2 test passed")