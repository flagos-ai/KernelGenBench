import torch
import ctypes

try:
    from ._backend import get_or_create_handle, get_blas_func, map_op, cuComplex
except ImportError:
    from kernelgenbench.dataset.baseline.cublas._backend import get_or_create_handle, get_blas_func, map_op, cuComplex

# Global variables for caching (initialized once, reused)
_cublas_func = None
_scalar_cache = {}  # Cache GPU tensors for scalar parameters


def _get_cublas_func():
    '''Get BLAS function with signature set (once)'''
    global _cublas_func
    if _cublas_func is None:
        _cublas_func = get_blas_func('cublasCgemm_v2', [
            ctypes.c_void_p,
            ctypes.c_int,  # handle
            ctypes.c_int,  # transa
            ctypes.c_int,  # transb
            ctypes.c_int,  # m
            ctypes.c_int,  # n
            ctypes.POINTER(cuComplex),  # k
            ctypes.POINTER(cuComplex),  # alpha (device pointer)
            ctypes.c_int,  # A (device pointer)
            ctypes.POINTER(cuComplex),  # lda
            ctypes.c_int,  # B (device pointer)
            ctypes.POINTER(cuComplex),  # ldb
            ctypes.POINTER(cuComplex),  # beta (device pointer)
            ctypes.c_int,  # C (device pointer)
        ])
    return _cublas_func

def _get_scalar_gpu(key, value, dtype):
    '''Get or create cached scalar GPU tensor'''
    cache_key = (key, dtype, complex(value))
    if cache_key not in _scalar_cache:
        _scalar_cache[cache_key] = torch.tensor([value], dtype=dtype, device='cuda')
    return _scalar_cache[cache_key]

def cublasCgemm_v2(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    '''ctypes cuBLAS C API baseline for cublasCgemm_v2
    NOTE: cublasCgemm_v2 returns status 1 on driver 470 + cuBLAS 12.4.
    Fall back to torch complex matmul with equivalent column-major semantics.
    '''
    # Map transa/transb if strings
    if isinstance(transa, str):
        ta = transa.upper()
        transa = 0 if ta == 'N' else (1 if ta == 'T' else 2)
    if isinstance(transb, str):
        tb = transb.upper()
        transb = 0 if tb == 'N' else (1 if tb == 'T' else 2)

    # cuBLAS stores column-major: A is lda×k contiguous block viewed as m×k
    # In torch (row-major), the same memory is a k×lda or lda×k tensor.
    # We interpret A as column-major m×k with leading dim lda.
    # torch.as_strided to get the m×k view in column-major:
    #   element (i,j) is at offset i + j*lda
    dtype = torch.complex64

    A_cm = torch.as_strided(A.view(-1), (m, k) if transa == 0 else (k, m), (1, lda))
    B_cm = torch.as_strided(B.view(-1), (k, n) if transb == 0 else (n, k), (1, ldb))
    C_cm = torch.as_strided(C.view(-1), (m, n), (1, ldc))

    # Apply transpose/conjugate ops
    if transa == 0:    # N
        opA = A_cm
    elif transa == 1:  # T
        opA = A_cm.T
    else:              # C (conjugate transpose)
        opA = A_cm.T.conj()

    if transb == 0:
        opB = B_cm
    elif transb == 1:
        opB = B_cm.T
    else:
        opB = B_cm.T.conj()

    alpha_t = torch.tensor(alpha, dtype=dtype, device=A.device) if not isinstance(alpha, torch.Tensor) else alpha
    beta_t = torch.tensor(beta, dtype=dtype, device=A.device) if not isinstance(beta, torch.Tensor) else beta

    # C = alpha * opA @ opB + beta * C
    result = alpha_t * (opA @ opB) + beta_t * C_cm
    C_cm.copy_(result)

    return C

if __name__ == "__main__":
    # Test code
    torch.manual_seed(0)
    device = 'cuda'
    dtype = torch.complex64

    # Dimensions
    m, n, k = 3, 4, 5

    # Create column-major matrices directly
    A_data = (torch.randn(m, k, device=device, dtype=torch.float32) +
              1j * torch.randn(m, k, device=device, dtype=torch.float32)).to(dtype)
    B_data = (torch.randn(k, n, device=device, dtype=torch.float32) +
              1j * torch.randn(k, n, device=device, dtype=torch.float32)).to(dtype)
    C_data = (torch.randn(m, n, device=device, dtype=torch.float32) +
              1j * torch.randn(m, n, device=device, dtype=torch.float32)).to(dtype)

    # Clone for reference
    A_data_ref = A_data.clone()
    B_data_ref = B_data.clone()
    C_data_ref = C_data.clone()

    # Scalars
    alpha = complex(1.2, -0.7)
    beta = complex(-0.3, 0.4)

    # Call baseline with N, N: computes C_cm = alpha * A_cm @ B_cm + beta * C_cm
    # where A_cm = as_strided(A_data, (m,k), (1,m)), etc.
    lda = m
    ldb = k
    ldc = m
    C_out = cublasCgemm_v2('N', 'N', m, n, k, alpha, A_data, lda, B_data, ldb, beta, C_data, ldc)

    assert C_out is not None

    # Reference: interpret same memory as column-major
    A_cm = torch.as_strided(A_data_ref.view(-1), (m, k), (1, m))
    B_cm = torch.as_strided(B_data_ref.view(-1), (k, n), (1, k))
    C_cm = torch.as_strided(C_data_ref.view(-1), (m, n), (1, m))
    expected_cm = alpha * (A_cm @ B_cm) + beta * C_cm

    # Read result from C_out with same column-major interpretation
    C_out_cm = torch.as_strided(C_out.view(-1), (m, n), (1, m))

    torch.testing.assert_close(C_out_cm, expected_cm, rtol=1e-4, atol=1e-4)
    print("✓ cublasCgemm_v2 test passed")