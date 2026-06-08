import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_op, cuDoubleComplex
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_op, cuDoubleComplex

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasZgemvBatched', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # trans
            ctypes.c_int,  # m
            ctypes.POINTER(cuDoubleComplex),  # n
            ctypes.POINTER(ctypes.c_void_p),  # alpha (device)
            ctypes.c_int,  # Aarray (device pointer array)
            ctypes.POINTER(ctypes.c_void_p),  # lda
            ctypes.c_int,  # xarray (device pointer array)
            ctypes.POINTER(cuDoubleComplex),  # incx
            ctypes.POINTER(ctypes.c_void_p),  # beta (device)
            ctypes.c_int,  # yarray (device pointer array)
            ctypes.c_int,  # incy
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

def cublasZgemvBatched(trans, m, n, alpha, Aarray, lda, xarray, incx, beta, yarray, incy, batchCount):
    '''ctypes cuBLAS C API baseline for cublasZgemvBatched'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    # Map trans if provided as string
    if isinstance(trans, str):
        if trans == 'N':
            trans = 0
        elif trans == 'T':
            trans = 1
        else:
            trans = 2  # 'C'
    trans = map_op(trans)

    # Aarray/xarray/yarray are int64 tensors on GPU holding device pointers
    Aarray_ptr = ctypes.c_void_p(Aarray.data_ptr())
    xarray_ptr = ctypes.c_void_p(xarray.data_ptr())
    yarray_ptr = ctypes.c_void_p(yarray.data_ptr())

    # Get cached scalar GPU tensors for alpha and beta (complex128)
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.complex128)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.complex128)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(trans),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.cast(Aarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(lda),
        ctypes.cast(xarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(incx),
        ctypes.cast(beta_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.cast(yarray_ptr, ctypes.POINTER(ctypes.c_void_p)),
        ctypes.c_int(incy),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasZgemvBatched failed with status {status}")
    return yarray

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)

    batchCount = 3
    m, n = 4, 5
    alpha = 0.7 + 0.2j
    beta = -0.3 + 0.5j

    A_list = []
    x_list = []
    y_list = []
    y_list_orig = []

    for _ in range(batchCount):
        A = torch.randn(m, n, device='cuda', dtype=torch.complex128).contiguous()
        x = torch.randn(n, device='cuda', dtype=torch.complex128).contiguous()
        y = torch.randn(m, device='cuda', dtype=torch.complex128).contiguous()

        A_list.append(A)
        x_list.append(x)
        y_list.append(y)
        y_list_orig.append(y.clone())

    A_ptrs = torch.tensor([t.data_ptr() for t in A_list], dtype=torch.int64, device='cuda')
    x_ptrs = torch.tensor([t.data_ptr() for t in x_list], dtype=torch.int64, device='cuda')
    y_ptrs = torch.tensor([t.data_ptr() for t in y_list], dtype=torch.int64, device='cuda')

    # Column-major trick for gemv:
    # Row-major A(m,n) is read as column-major (n,m) by cuBLAS.
    # We want y = alpha * A * x + beta * y, i.e. (m,n)*(n,) -> (m,)
    # cuBLAS sees A_cm as (n,m). With trans='T': op(A_cm) = A_cm^T = (m,n)
    # cuBLAS m_param=n, n_param=m, lda=n
    lda = n
    incx = 1
    incy = 1

    result_ptrs = cublasZgemvBatched('T', n, m, alpha, A_ptrs, lda, x_ptrs, incx, beta, y_ptrs, incy, batchCount)
    assert result_ptrs is not None

    expected_list = [alpha * (A_list[i] @ x_list[i]) + beta * y_list_orig[i] for i in range(batchCount)]

    y_stacked = torch.stack(y_list, dim=0)
    expected_stacked = torch.stack(expected_list, dim=0)

    torch.testing.assert_close(y_stacked, expected_stacked, rtol=1e-5, atol=1e-5)
    print("✓ cublasZgemvBatched test passed")