import torch
import ctypes

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_scalar, destroy_acl_scalar,
                           two_stage_launch, ACL_FLOAT, cuda_dtype_to_torch,
                           torch_dtype_to_acl)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_scalar, destroy_acl_scalar,
        two_stage_launch, ACL_FLOAT, cuda_dtype_to_torch,
        torch_dtype_to_acl)


def cublasSgemmEx(transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc):
    """CANN baseline for cublasSgemmEx using ctypes aclnnGemm C API.

    Mixed-precision gemm: C = alpha * op(A) @ op(B) + beta * C
    Atype, Btype, Ctype are cudaDataType enums.
    Data is in column-major layout.
    """
    if isinstance(transa, str):
        transa_int = 0 if transa.upper() == 'N' else 1
    else:
        transa_int = 1 if transa in (1, 2) else 0
    if isinstance(transb, str):
        transb_int = 0 if transb.upper() == 'N' else 1
    else:
        transb_int = 1 if transb in (1, 2) else 0

    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)
    C_flat = C.reshape(-1)

    A_cm = torch.as_strided(A_flat, (m, k) if transa_int == 0 else (k, m), (1, lda))
    B_cm = torch.as_strided(B_flat, (k, n) if transb_int == 0 else (n, k), (1, ldb))
    C_cm = torch.as_strided(C_flat, (m, n), (1, ldc))

    A_cont = A_cm.contiguous()
    B_cont = B_cm.contiguous()
    C_cont = C_cm.contiguous()

    A_t = create_acl_tensor(A_cont, list(A_cont.shape), list(A_cont.stride()))
    B_t = create_acl_tensor(B_cont, list(B_cont.shape), list(B_cont.stride()))
    C_t = create_acl_tensor(C_cont, list(C_cont.shape), list(C_cont.stride()))
    out_t = create_acl_tensor(C_cont, list(C_cont.shape), list(C_cont.stride()))

    # Use float32 for scalars in mixed-precision gemm
    alpha_s = create_acl_scalar(alpha, ACL_FLOAT)
    beta_s = create_acl_scalar(beta, ACL_FLOAT)

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

    CUDA_R_32F = 0
    m, n, k = 64, 48, 32
    alpha, beta = 1.5, -0.75

    A = torch.randn(m, k, dtype=torch.float32, device=device)
    B = torch.randn(k, n, dtype=torch.float32, device=device)
    C = torch.randn(m, n, dtype=torch.float32, device=device)
    A0, B0, C0 = A.clone(), B.clone(), C.clone()

    At = A.t().contiguous()
    Bt = B.t().contiguous()
    Ct = C.t().contiguous()
    lda, ldb, ldc = m, k, m

    result = cublasSgemmEx('N', 'N', m, n, k, alpha, At, CUDA_R_32F, lda, Bt, CUDA_R_32F, ldb, beta, Ct, CUDA_R_32F, ldc)
    expected = alpha * (A0 @ B0) + beta * C0
    result_rm = result.t()
    torch.testing.assert_close(result_rm, expected, rtol=1e-5, atol=1e-5)
    print("\n✓ cublasSgemmEx all tests passed")
