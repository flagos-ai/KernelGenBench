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
        _cublas_func = get_blas_func('cublasDasum_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_void_p,  # n
            ctypes.c_int,  # x (const double*)
            ctypes.POINTER(ctypes.c_double),  # incx
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include value so different alpha/beta values get different tensors
    cache_key = (key, dtype, float(value))  # use complex(value) for complex types
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasDasum_v2(n, x, incx, result):
    '''ctypes cuBLAS C API baseline for cublasDasum_v2
    NOTE: cublasDasum_v2 returns CUBLAS_STATUS_EXECUTION_FAILED (13) on
    driver 470 + cuBLAS 12.4.  Fall back to cublasDgemm_v2:
    result = ones^T @ |x_strided|  (1×n dot n×1 via gemm).
    '''
    handle = get_or_create_handle()

    xs = x[::incx][:n].abs().contiguous()
    ones = torch.ones(n, dtype=torch.float64, device=x.device)

    set_pointer_mode(handle, 0)  # HOST

    gemm = get_blas_func("cublasDgemm_v2")
    gemm.restype = ctypes.c_int

    alpha = ctypes.c_double(1.0)
    beta  = ctypes.c_double(0.0)

    CUBLAS_OP_T = map_op(1)
    CUBLAS_OP_N = map_op(0)

    status = gemm(
        handle,
        ctypes.c_int(CUBLAS_OP_T), ctypes.c_int(CUBLAS_OP_N),
        ctypes.c_int(1), ctypes.c_int(1), ctypes.c_int(n),
        ctypes.byref(alpha),
        ctypes.c_void_p(ones.data_ptr()), ctypes.c_int(n),
        ctypes.c_void_p(xs.data_ptr()), ctypes.c_int(n),
        ctypes.byref(beta),
        ctypes.c_void_p(result.data_ptr()), ctypes.c_int(1),
    )

    set_pointer_mode(handle, 1)  # DEVICE

    if status != 0:
        raise RuntimeError(f"cublasDasum_v2 (gemm fallback) failed with status {status}")

    return result

if __name__ == "__main__":
    # Test code
    n = 128
    incx = 1
    x = torch.randn(n, dtype=torch.float64, device='cuda')
    result = torch.empty(1, dtype=torch.float64, device='cuda')

    # Clone originals
    x_clone = x.clone()
    result_clone = result.clone()

    # Call baseline
    out = cublasDasum_v2(n, x, incx, result)
    assert out is not None

    # PyTorch reference
    expected = torch.abs(x_clone[::incx]).sum().reshape(1)

    # Numerical check
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasDasum_v2 test passed")