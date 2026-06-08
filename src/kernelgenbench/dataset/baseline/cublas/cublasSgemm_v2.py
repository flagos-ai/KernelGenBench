import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_op
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_op

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters

def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasSgemm_v2', [
            ctypes.c_void_p,            # handle
            ctypes.c_int,               # transa
            ctypes.c_int,               # transb
            ctypes.c_int,               # m
            ctypes.c_int,               # n
            ctypes.c_int,               # k
            ctypes.POINTER(ctypes.c_float),  # alpha (device ptr)
            ctypes.POINTER(ctypes.c_float),  # A (device ptr)
            ctypes.c_int,               # lda
            ctypes.POINTER(ctypes.c_float),  # B (device ptr)
            ctypes.c_int,               # ldb
            ctypes.POINTER(ctypes.c_float),  # beta (device ptr)
            ctypes.POINTER(ctypes.c_float),  # C (device ptr)
            ctypes.c_int                # ldc
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasSgemm_v2(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    '''ctypes BLAS C API baseline for cublasSgemm_v2'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    # Map operation enum to backend (cuBLAS: 0/1, hipBLAS: 111/112)
    transa = map_op(transa)
    transb = map_op(transb)

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    B_ptr = ctypes.c_void_p(B.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensors
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    # Call BLAS C API
    status = func(
        handle,
        ctypes.c_int(transa),
        ctypes.c_int(transb),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(A_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(lda),
        ctypes.cast(B_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(ldb),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(C_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(ldc)
    )
    if status != 0:
        raise RuntimeError(f"cublasSgemm_v2 failed with status {status}")
    return C

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)

    m, n, k = 64, 48, 32

    A = torch.randn(m, k, dtype=torch.float32, device='cuda')
    B = torch.randn(k, n, dtype=torch.float32, device='cuda')
    C = torch.randn(m, n, dtype=torch.float32, device='cuda')

    A0 = A.clone()
    B0 = B.clone()
    C0 = C.clone()

    alpha = 1.5
    beta = 0.75

    result = cublasSgemm_v2(
        'N', 'N',
        n, m, k,
        alpha,
        B, n,
        A, k,
        beta,
        C, n
    )

    assert result is not None

    expected = alpha * (A0 @ B0) + beta * C0
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasSgemm_v2 test passed")
