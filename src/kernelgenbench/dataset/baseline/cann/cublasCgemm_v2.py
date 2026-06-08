import torch
import ctypes

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_scalar, destroy_acl_scalar,
                           two_stage_launch, ACL_COMPLEX64)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_scalar, destroy_acl_scalar,
        two_stage_launch, ACL_COMPLEX64)


def cublasCgemm_v2(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """CANN baseline for cublasCgemm_v2 using ctypes aclnnGemm C API.

    Computes: C = alpha * op(A) @ op(B) + beta * C (complex64)
    Data is in column-major layout.
    """
    if isinstance(transa, str):
        transa_int = {'N': 0, 'T': 1, 'C': 2}.get(transa.upper(), 0)
    else:
        transa_int = transa
    if isinstance(transb, str):
        transb_int = {'N': 0, 'T': 1, 'C': 2}.get(transb.upper(), 0)
    else:
        transb_int = transb

    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)
    C_flat = C.reshape(-1)

    A_cm = torch.as_strided(A_flat, (m, k) if transa_int == 0 else (k, m), (1, lda))
    B_cm = torch.as_strided(B_flat, (k, n) if transb_int == 0 else (n, k), (1, ldb))
    C_cm = torch.as_strided(C_flat, (m, n), (1, ldc))

    # Handle conjugate: resolve before making contiguous
    if transa_int == 2:
        A_cont = A_cm.conj().resolve_conj().contiguous()
        transa_int = 1  # treat as transpose after conjugation
    else:
        A_cont = A_cm.contiguous()
    if transb_int == 2:
        B_cont = B_cm.conj().resolve_conj().contiguous()
        transb_int = 1
    else:
        B_cont = B_cm.contiguous()
    C_cont = C_cm.contiguous()

    A_t = create_acl_tensor(A_cont, list(A_cont.shape), list(A_cont.stride()))
    B_t = create_acl_tensor(B_cont, list(B_cont.shape), list(B_cont.stride()))
    C_t = create_acl_tensor(C_cont, list(C_cont.shape), list(C_cont.stride()))
    out_t = create_acl_tensor(C_cont, list(C_cont.shape), list(C_cont.stride()))

    alpha_s = create_acl_scalar(complex(alpha), ACL_COMPLEX64)
    beta_s = create_acl_scalar(complex(beta), ACL_COMPLEX64)

    two_stage_launch('aclnnGemmGetWorkspaceSize', 'aclnnGemm',
                     [A_t, B_t, C_t, alpha_s, beta_s,
                      ctypes.c_int64(transa_int), ctypes.c_int64(transb_int),
                      out_t, ctypes.c_int8(0)])

    destroy_acl_tensor(A_t)
    destroy_acl_tensor(B_t)
    destroy_acl_tensor(C_t)
    destroy_acl_tensor(out_t)
    destroy_acl_scalar(alpha_s)
    destroy_acl_scalar(beta_s)

    C_cm.copy_(C_cont)
    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'
    dtype = torch.complex64

    m, n, k = 3, 4, 5
    alpha = complex(1.2, -0.7)
    beta = complex(-0.3, 0.4)

    A_rm = torch.randn(m, k, dtype=dtype, device=device)
    B_rm = torch.randn(k, n, dtype=dtype, device=device)
    C_rm = torch.randn(m, n, dtype=dtype, device=device)

    A_cm = A_rm.t().contiguous()
    B_cm = B_rm.t().contiguous()
    C_cm = C_rm.t().contiguous()

    lda, ldb, ldc = m, k, m

    result = cublasCgemm_v2('N', 'N', m, n, k, alpha, A_cm, lda, B_cm, ldb, beta, C_cm, ldc)
    expected = alpha * (A_rm @ B_rm) + beta * C_rm
    result_rm = torch.as_strided(result.reshape(-1), (m, n), (1, m)).clone()
    torch.testing.assert_close(result_rm, expected, rtol=1e-4, atol=1e-4)
    print("pass: N, N")

    print("\n✓ cublasCgemm_v2 all tests passed")
