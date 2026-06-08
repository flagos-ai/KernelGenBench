import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, set_pointer_mode, map_op, cuComplex
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, set_pointer_mode, map_op, cuComplex

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasCdotu_v2', [
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
    val_key = complex(value) if dtype.is_complex else float(value)
    cache_key = (key, dtype, val_key)
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasCdotu_v2(n, x, incx, y, incy, result):
    '''ctypes cuBLAS C API baseline for cublasCdotu_v2
    NOTE: cublasCdotu_v2 returns status 15 on driver 470 + cuBLAS 12.4.
    Fall back to cublasCgemm_v2:  result(1×1) = x^T · y  (unconjugated).
    '''
    handle = get_or_create_handle()

    xs = x[::incx][:n].contiguous()
    ys = y[::incy][:n].contiguous()

    set_pointer_mode(handle, 0)  # HOST

    gemm = get_blas_func("cublasCgemm_v2")
    gemm.restype = ctypes.c_int

    alpha = cuComplex(1.0, 0.0)
    beta  = cuComplex(0.0, 0.0)

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

    set_pointer_mode(handle, 1)  # DEVICE

    if status != 0:
        raise RuntimeError(f"cublasCdotu_v2 (gemm fallback) failed with status {status}")

    return result

if __name__ == "__main__":
    # Test code
    n = 16
    x = torch.randn(n, dtype=torch.float32, device='cuda') + 1j * torch.randn(n, dtype=torch.float32, device='cuda')
    y = torch.randn(n, dtype=torch.float32, device='cuda') + 1j * torch.randn(n, dtype=torch.float32, device='cuda')
    x = x.to(torch.complex64)
    y = y.to(torch.complex64)
    x0 = x.clone()
    y0 = y.clone()

    result = torch.empty(1, dtype=torch.complex64, device='cuda')
    out = cublasCdotu_v2(n, x, 1, y, 1, result)
    assert out is not None

    expected_val = (x0 * y0).sum()
    expected = expected_val.unsqueeze(0)

    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasCdotu_v2 test passed")