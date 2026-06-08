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
        _cublas_func = get_blas_func('cublasZgemmBatched', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # transa
            ctypes.c_int,  # transb
            ctypes.c_int,  # m
            ctypes.c_int,  # n
            ctypes.POINTER(cuDoubleComplex),  # k
            ctypes.POINTER(ctypes.c_void_p),  # alpha (device)
            ctypes.c_int,  # Aarray (device pointer array)
            ctypes.POINTER(ctypes.c_void_p),  # lda
            ctypes.c_int,  # Barray (device pointer array)
            ctypes.POINTER(cuDoubleComplex),  # ldb
            ctypes.POINTER(ctypes.c_void_p),  # beta (device)
            ctypes.c_int,  # Carray (device pointer array)
            ctypes.c_int,  # ldc
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

def cublasZgemmBatched(transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount):
    '''ctypes cuBLAS C API baseline for cublasZgemmBatched'''
    handle = get_or_create_handle()
    func = _get_cublas_func()


    # Map operation enum to backend (cuBLAS: 0/1, hipBLAS: 111/112)
    transa = map_op(transa)
    transb = map_op(transb)
    # Map transposition parameters if provided as characters
    if isinstance(transa, str):
        transa = 0 if transa == 'N' else (1 if transa == 'T' else 2)
    if isinstance(transb, str):
        transb = 0 if transb == 'N' else (1 if transb == 'T' else 2)

    # Aarray/Barray/Carray are int64 tensors on GPU holding device pointers
    Aarray_ptr = ctypes.c_void_p(Aarray.data_ptr())
    Barray_ptr = ctypes.c_void_p(Barray.data_ptr())
    Carray_ptr = ctypes.c_void_p(Carray.data_ptr())

    # Get cached scalar GPU tensors (complex128)
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.complex128)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.complex128)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(transa), ctypes.c_int(transb),
        ctypes.c_int(m), ctypes.c_int(n), ctypes.c_int(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.cast(Aarray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int(lda),
        ctypes.cast(Barray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int(ldb),
        ctypes.cast(beta_ptr, ctypes.POINTER(cuDoubleComplex)),
        ctypes.cast(Carray_ptr, ctypes.POINTER(ctypes.c_void_p)), ctypes.c_int(ldc),
        ctypes.c_int(batchCount)
    )
    if status != 0:
        raise RuntimeError(f"cublasZgemmBatched failed with status {status}")
    return Carray

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)

    # Dimensions
    batchCount = 3
    m, n, k = 4, 5, 3

    # Create test matrices (row-major) on GPU
    A_list = [torch.randn(m, k, dtype=torch.complex128, device='cuda') +
              1j * torch.randn(m, k, dtype=torch.complex128, device='cuda') for _ in range(batchCount)]
    B_list = [torch.randn(k, n, dtype=torch.complex128, device='cuda') +
              1j * torch.randn(k, n, dtype=torch.complex128, device='cuda') for _ in range(batchCount)]
    C_list = [torch.randn(m, n, dtype=torch.complex128, device='cuda') +
              1j * torch.randn(m, n, dtype=torch.complex128, device='cuda') for _ in range(batchCount)]

    # Clone originals for expected computation
    A_ref = [a.clone() for a in A_list]
    B_ref = [b.clone() for b in B_list]
    C_ref = [c.clone() for c in C_list]

    # Convert to column-major compatible buffers using transposed contiguous tensors
    A_cm = [a.transpose(-2, -1).contiguous() for a in A_list]  # represents column-major (m,k)
    B_cm = [b.transpose(-2, -1).contiguous() for b in B_list]  # represents column-major (k,n)
    C_cm = [c.transpose(-2, -1).contiguous() for c in C_list]  # represents column-major (m,n)

    # Build device pointer arrays (int64 tensors on GPU)
    A_ptrs = torch.tensor([t.data_ptr() for t in A_cm], dtype=torch.int64, device='cuda')
    B_ptrs = torch.tensor([t.data_ptr() for t in B_cm], dtype=torch.int64, device='cuda')
    C_ptrs = torch.tensor([t.data_ptr() for t in C_cm], dtype=torch.int64, device='cuda')

    # Scalars
    alpha = 0.7 + 0.2j
    beta = -0.3 + 0.5j

    # Leading dimensions for column-major views
    lda = m
    ldb = k
    ldc = m

    # Call baseline
    out_ptrs = cublasZgemmBatched('N', 'N', m, n, k, alpha, A_ptrs, lda, B_ptrs, ldb, beta, C_ptrs, ldc, batchCount)
    assert out_ptrs is not None

    # Compute expected result in PyTorch (row-major)
    expected_list = [alpha * (A_ref[i] @ B_ref[i]) + beta * C_ref[i] for i in range(batchCount)]

    # Gather results from C_cm and convert back to row-major
    result_list = [C_cm[i].transpose(-2, -1).contiguous() for i in range(batchCount)]

    # Numerical check
    for i in range(batchCount):
        torch.testing.assert_close(result_list[i], expected_list[i], rtol=1e-5, atol=1e-5)

    print("✓ cublasZgemmBatched test passed")