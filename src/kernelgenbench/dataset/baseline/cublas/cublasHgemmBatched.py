import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_op
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_op

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters

# Define half type for ctypes
c_half = ctypes.c_uint16


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasHgemmBatched', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # transa
            ctypes.c_int,  # transb
            ctypes.c_int,  # m
            ctypes.c_int,  # n
            ctypes.POINTER(c_half),  # k
            ctypes.POINTER(ctypes.c_void_p),  # alpha (device)
            ctypes.c_int,  # Aarray (device pointer array)
            ctypes.POINTER(ctypes.c_void_p),  # lda
            ctypes.c_int,  # Barray (device pointer array)
            ctypes.POINTER(c_half),  # ldb
            ctypes.POINTER(ctypes.c_void_p),  # beta (device)
            ctypes.c_int,  # Carray (device pointer array)
            ctypes.c_int,  # ldc
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    # CRITICAL: cache key must include value so different alpha/beta values get different tensors
    cache_key = (key, dtype, float(value))  # use complex(value) for complex types
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasHgemmBatched(transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount):
    '''ctypes cuBLAS C API baseline for cublasHgemmBatched'''
    handle = get_or_create_handle()
    func = _get_cublas_func()


    # Map operation enum to backend (cuBLAS: 0/1, hipBLAS: 111/112)
    transa = map_op(transa)
    transb = map_op(transb)
    # Map transa/transb if provided as characters
    if isinstance(transa, str):
        transa = 0 if transa == 'N' else (1 if transa == 'T' else 2)
    if isinstance(transb, str):
        transb = 0 if transb == 'N' else (1 if transb == 'T' else 2)

    # Aarray/Barray/Carray are int64 tensors on GPU holding device pointers
    Aarray_ptr = ctypes.c_void_p(Aarray.data_ptr())
    Barray_ptr = ctypes.c_void_p(Barray.data_ptr())
    Carray_ptr = ctypes.c_void_p(Carray.data_ptr())

    # Get cached scalar GPU tensors for alpha/beta (__half)
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float16)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float16)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(transa), ctypes.c_int(transb),
        ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(c_half)),
        ctypes.cast(Aarray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int(lda),
        ctypes.cast(Barray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int(ldb),
        ctypes.cast(beta_ptr, ctypes.POINTER(c_half)),
        ctypes.cast(Carray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int(ldc),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasHgemmBatched failed with status {status}")
    return Carray

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    device = 'cuda'

    # Dimensions
    m, n, k = 4, 5, 6
    batchCount = 3

    # Create test matrices (row-major), small integer range to ensure exactness in fp16
    A_list = [torch.randint(-2, 3, (m, k), dtype=torch.int8, device=device).to(torch.float16) for _ in range(batchCount)]
    B_list = [torch.randint(-2, 3, (k, n), dtype=torch.int8, device=device).to(torch.float16) for _ in range(batchCount)]
    C_list = [torch.zeros((m, n), dtype=torch.float16, device=device) for _ in range(batchCount)]  # not used directly

    # Convert to column-major friendly buffers by using transposed, contiguous tensors
    A_f_list = [A.t().contiguous() for A in A_list]          # shape (k, m) but used as col-major (m, k)
    B_f_list = [B.t().contiguous() for B in B_list]          # shape (n, k) but used as col-major (k, n)
    C_f_list = [torch.zeros((n, m), dtype=torch.float16, device=device) for _ in range(batchCount)]  # used as col-major (m, n)

    # Build device arrays of pointers (int64 tensor on GPU)
    A_ptrs = torch.tensor([t.data_ptr() for t in A_f_list], dtype=torch.int64).to(device)
    B_ptrs = torch.tensor([t.data_ptr() for t in B_f_list], dtype=torch.int64).to(device)
    C_ptrs = torch.tensor([t.data_ptr() for t in C_f_list], dtype=torch.int64).to(device)

    # Scalars
    alpha = 1.0
    beta = 0.0

    # Leading dimensions for column-major (rows of each matrix)
    lda = m
    ldb = k
    ldc = m

    # Clone originals for comparison
    A_f_clone = [t.clone() for t in A_f_list]
    B_f_clone = [t.clone() for t in B_f_list]
    C_f_clone = [t.clone() for t in C_f_list]

    # Call baseline
    ret = cublasHgemmBatched('N', 'N', m, n, k, alpha, A_ptrs, lda, B_ptrs, ldb, beta, C_ptrs, ldc, batchCount)
    assert ret is not None

    # PyTorch reference: expected_f = (A @ B).t() to match our column-major buffers (stored as (n, m))
    expected_f_list = []
    for i in range(batchCount):
        A_rm = A_f_clone[i].t().contiguous()  # back to row-major A (m, k)
        B_rm = B_f_clone[i].t().contiguous()  # back to row-major B (k, n)
        expected = A_rm @ B_rm                # (m, n)
        expected_f_list.append(expected.t().contiguous())  # (n, m), matches C_f layout

    # Stack for numerical check
    result = torch.stack(C_f_list, dim=0)
    expected = torch.stack(expected_f_list, dim=0)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasHgemmBatched test passed")