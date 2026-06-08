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
        _cublas_func = get_blas_func('cublasDgemvBatched', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # trans
            ctypes.c_int,  # m
            ctypes.POINTER(ctypes.c_double),  # n
            ctypes.POINTER(ctypes.c_void_p),  # alpha (device)
            ctypes.c_int,  # Aarray (device pointer array)
            ctypes.POINTER(ctypes.c_void_p),  # lda
            ctypes.c_int,  # xarray (device pointer array)
            ctypes.POINTER(ctypes.c_double),  # incx
            ctypes.POINTER(ctypes.c_void_p),  # beta (device)
            ctypes.c_int,  # yarray (device pointer array)
            ctypes.c_int,  # incy
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include value so different alpha/beta values get different tensors
    cache_key = (key, dtype, float(value))  # use complex(value) for complex types
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasDgemvBatched(trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount):
    '''ctypes cuBLAS C API baseline for cublasDgemvBatched'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    if isinstance(trans, str):
        trans = 0 if trans == 'N' else (1 if trans == 'T' else 2)
        trans = map_op(trans)

    # Aarray/xarray/yarray are int64 tensors on GPU holding device pointers
    Aarray_ptr = ctypes.c_void_p(Aarray.data_ptr())
    xarray_ptr = ctypes.c_void_p(xarray.data_ptr())
    yarray_ptr = ctypes.c_void_p(yarray.data_ptr())

    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float64)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float64)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(trans),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(Aarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(lda),
        ctypes.cast(xarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(incx),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_double)),
        ctypes.cast(yarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(incy),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasDgemvBatched failed with status {status}")
    return yarray

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    batchCount = 3
    m, n = 4, 5
    alpha = 1.7
    beta = 0.3
    incx = 1
    incy = 1

    # Create batched matrices and vectors on GPU (double precision)
    A_list = [torch.randn(m, n, dtype=torch.float64, device='cuda').contiguous() for _ in range(batchCount)]
    x_list = [torch.randn(n, dtype=torch.float64, device='cuda').contiguous() for _ in range(batchCount)]
    y_list = [torch.randn(m, dtype=torch.float64, device='cuda').contiguous() for _ in range(batchCount)]

    A_list_ref = [A.clone() for A in A_list]
    x_list_ref = [x.clone() for x in x_list]
    y_list_orig = [y.clone() for y in y_list]

    # Build device pointer arrays
    Aarray = torch.tensor([A.data_ptr() for A in A_list], dtype=torch.int64, device='cuda')
    xarray = torch.tensor([x.data_ptr() for x in x_list], dtype=torch.int64, device='cuda')
    yarray = torch.tensor([y.data_ptr() for y in y_list], dtype=torch.int64, device='cuda')

    # Column-major trick for gemv:
    # Row-major A(m,n) is read as column-major (n,m) by cuBLAS.
    # We want y = alpha * A * x + beta * y, i.e. (m,n)*(n,) -> (m,)
    # cuBLAS sees A_cm as (n,m). With trans='T': op(A_cm) = A_cm^T = (m,n), then y(m) = alpha*(m,n)*x(n)+beta*y(m) ✓
    # cuBLAS m_param=n (rows of A_cm), n_param=m (cols of A_cm), lda=n
    result_ptrs = cublasDgemvBatched('T', n, m, alpha, Aarray, n, xarray, incx, beta, yarray, incy, batchCount)
    assert result_ptrs is not None

    expected_list = [alpha * (A_list_ref[i] @ x_list_ref[i]) + beta * y_list_orig[i] for i in range(batchCount)]

    result = torch.stack(y_list, dim=0)
    expected = torch.stack(expected_list, dim=0)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasDgemvBatched test passed")