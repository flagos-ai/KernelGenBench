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
        _cublas_func = get_blas_func('cublasSgemmBatched_64', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # transa
            ctypes.c_int64,  # transb
            ctypes.c_int64,  # m
            ctypes.c_int64,  # n
            ctypes.POINTER(ctypes.c_float),  # k
            ctypes.POINTER(ctypes.c_void_p),  # alpha (device)
            ctypes.c_int64,  # Aarray (device pointer array)
            ctypes.POINTER(ctypes.c_void_p),  # lda
            ctypes.c_int64,  # Barray (device pointer array)
            ctypes.POINTER(ctypes.c_float),  # ldb
            ctypes.POINTER(ctypes.c_void_p),  # beta (device)
            ctypes.c_int64,  # Carray (device pointer array)
            ctypes.c_int64,  # ldc
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include value so different alpha/beta values get different tensors
    cache_key = (key, dtype, float(value))  # use complex(value) for complex types
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasSgemmBatched_64(transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount):
    '''ctypes cuBLAS C API baseline for cublasSgemmBatched_64'''
    handle = get_or_create_handle()
    func = _get_cublas_func()


    # Map operation enum to backend (cuBLAS: 0/1, hipBLAS: 111/112)
    transa = map_op(transa)
    transb = map_op(transb)
    # Optional conversion if transa/transb are provided as characters
    if isinstance(transa, str):
        transa = 0 if transa == 'N' else (1 if transa == 'T' else 2)
    if isinstance(transb, str):
        transb = 0 if transb == 'N' else (1 if transb == 'T' else 2)

    # Aarray/Barray/Carray are int64 tensors on GPU holding device pointers
    Aarray_ptr = ctypes.c_void_p(Aarray.data_ptr())
    Barray_ptr = ctypes.c_void_p(Barray.data_ptr())
    Carray_ptr = ctypes.c_void_p(Carray.data_ptr())

    # Get cached scalar GPU tensors for alpha/beta
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(transa), ctypes.c_int(transb),
        ctypes.c_int64(m), ctypes.c_int64(n), ctypes.c_int64(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(Aarray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int64(lda),
        ctypes.cast(Barray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int64(ldb),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(Carray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int64(ldc),
        ctypes.c_int64(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasSgemmBatched_64 failed with status {status}")
    return Carray

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    device = 'cuda'

    # Dimensions
    m, n, k = 4, 5, 6
    batchCount = 3

    # Create batched input matrices on GPU (row-major)
    A_list = [torch.randn(m, k, dtype=torch.float32, device=device) for _ in range(batchCount)]
    B_list = [torch.randn(k, n, dtype=torch.float32, device=device) for _ in range(batchCount)]
    C_list = [torch.randn(m, n, dtype=torch.float32, device=device) for _ in range(batchCount)]

    # Clones for reference computation (row-major)
    A_ref = [a.clone() for a in A_list]
    B_ref = [b.clone() for b in B_list]
    C_ref = [c.clone() for c in C_list]

    # Build column-major representations by transposing and making contiguous
    # Row-major (m,k) -> Column-major (m,k) by using a.t().contiguous() with shape (k,m)
    A_cm = [a.t().contiguous() for a in A_list]  # treated as column-major (m,k)
    B_cm = [b.t().contiguous() for b in B_list]  # treated as column-major (k,n)
    C_cm = [c.t().contiguous() for c in C_list]  # treated as column-major (m,n)

    # Create device arrays of pointers (int64) to each matrix
    Aarray = torch.tensor([int(x.data_ptr()) for x in A_cm], dtype=torch.int64, device=device)
    Barray = torch.tensor([int(x.data_ptr()) for x in B_cm], dtype=torch.int64, device=device)
    Carray = torch.tensor([int(x.data_ptr()) for x in C_cm], dtype=torch.int64, device=device)

    # Scalars
    alpha = 1.5
    beta = -0.25

    # Leading dimensions for column-major: lda=m, ldb=k, ldc=m
    lda = m
    ldb = k
    ldc = m

    # Call baseline
    out = cublasSgemmBatched_64('N', 'N', m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)
    assert out is not None

    # Compute expected result in PyTorch (row-major)
    expected_list = []
    for i in range(batchCount):
        expected = alpha * (A_ref[i] @ B_ref[i]) + beta * C_ref[i]
        expected_list.append(expected)

    # Retrieve result from column-major buffers by transposing back
    result_list = [C_cm[i].t().contiguous() for i in range(batchCount)]

    # Numerical check
    for i in range(batchCount):
        torch.testing.assert_close(result_list[i], expected_list[i], rtol=1e-2, atol=1e-2)

    print("✓ cublasSgemmBatched_64 test passed")