import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_op, map_data_type
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_op, map_data_type

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasSgemmEx', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # transa
            ctypes.c_int,  # transb
            ctypes.c_int,  # m
            ctypes.c_int,  # n
            ctypes.POINTER(ctypes.c_float),  # k
            ctypes.c_void_p,  # alpha
            ctypes.c_int,  # A
            ctypes.c_int,  # Atype (cudaDataType)
            ctypes.c_void_p,  # lda
            ctypes.c_int,  # B
            ctypes.c_int,  # Btype (cudaDataType)
            ctypes.POINTER(ctypes.c_float),  # ldb
            ctypes.c_void_p,  # beta
            ctypes.c_int,  # C
            ctypes.c_int,  # Ctype (cudaDataType)
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, float(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasSgemmEx(transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc):
    '''ctypes cuBLAS C API baseline for cublasSgemmEx'''
    handle = get_or_create_handle()
    func = _get_cublas_func()

    # Map transposition parameters if passed as characters
    if isinstance(transa, str):
        transa = 0 if transa in ('N', 'n') else (1 if transa in ('T', 't') else 2)
    if isinstance(transb, str):
        transb = 0 if transb in ('N', 'n') else (1 if transb in ('T', 't') else 2)
    # Map operation enum to backend (cuBLAS: 0/1, hipBLAS: 111/112)
    transa = map_op(transa)
    transb = map_op(transb)

    # Map data type enums to backend (cuBLAS: 0, hipBLAS: 151)
    Atype = map_data_type(Atype)
    Btype = map_data_type(Btype)
    Ctype = map_data_type(Ctype)

    # Convert tensors to GPU pointers
    A_ptr = ctypes.c_void_p(A.data_ptr())
    B_ptr = ctypes.c_void_p(B.data_ptr())
    C_ptr = ctypes.c_void_p(C.data_ptr())

    # Get cached scalar GPU tensors for alpha and beta
    alpha_gpu = _get_scalar_gpu('alpha', alpha, torch.float32)
    beta_gpu = _get_scalar_gpu('beta', beta, torch.float32)
    alpha_ptr = ctypes.c_void_p(alpha_gpu.data_ptr())
    beta_ptr = ctypes.c_void_p(beta_gpu.data_ptr())

    status = func(
        handle,
        ctypes.c_int(transa),
        ctypes.c_int(transb),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.cast(alpha_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(A_ptr, ctypes.c_void_p),
        ctypes.c_int(Atype),
        ctypes.c_int(lda),
        ctypes.cast(B_ptr, ctypes.c_void_p),
        ctypes.c_int(Btype),
        ctypes.c_int(ldb),
        ctypes.cast(beta_ptr, ctypes.POINTER(ctypes.c_float)),
        ctypes.cast(C_ptr, ctypes.c_void_p),
        ctypes.c_int(Ctype),
        ctypes.c_int(ldc)
    )
    if status != 0:
        raise RuntimeError(f"cublasSgemmEx failed with status {status}")
    return C

if __name__ == "__main__":
    # Constants for cudaDataType
    CUDA_R_32F = 0

    # Test data
    m, n, k = 64, 48, 32
    alpha = 1.5
    beta = -0.75

    A = torch.randn(m, k, dtype=torch.float32, device='cuda')
    B = torch.randn(k, n, dtype=torch.float32, device='cuda')
    C = torch.randn(m, n, dtype=torch.float32, device='cuda')

    # Clone originals for reference
    A0 = A.clone()
    B0 = B.clone()
    C0 = C.clone()

    # Adjust for column-major cuBLAS expectations using transposed, contiguous tensors
    At = A.t().contiguous()  # shape (k, m)
    Bt = B.t().contiguous()  # shape (n, k)
    Ct = C.t().contiguous()  # shape (n, m)

    # Leading dimensions for column-major (match rows in column-major view)
    lda = m
    ldb = k
    ldc = m

    # Call baseline
    out = cublasSgemmEx(
        'N', 'N',
        m, n, k,
        alpha,
        At, CUDA_R_32F, lda,
        Bt, CUDA_R_32F, ldb,
        beta,
        Ct, CUDA_R_32F, ldc
    )

    assert out is not None

    # Compute PyTorch reference in row-major
    expected = alpha * (A0 @ B0) + beta * C0

    # Convert cuBLAS result back to row-major view
    result = Ct.t()

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasSgemmEx test passed")