import torch
import ctypes

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_scalar, destroy_acl_scalar,
                           two_stage_launch, ACL_FLOAT, map_transpose)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_scalar, destroy_acl_scalar,
        two_stage_launch, ACL_FLOAT, map_transpose)


def cublasSgeam(transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc):
    """CANN baseline for cublasSgeam using ctypes aclnn C API.

    Computes C = alpha * op(A) + beta * op(B) (float32)
    Data is in column-major layout.
    Uses aclnnMuls + aclnnAdd.
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

    # Column-major views
    A_cm = torch.as_strided(A_flat, (m, n) if transa_int == 0 else (n, m), (1, lda))
    B_cm = torch.as_strided(B_flat, (m, n) if transb_int == 0 else (n, m), (1, ldb))
    C_cm = torch.as_strided(C_flat, (m, n), (1, ldc))

    opA = (A_cm if transa_int == 0 else A_cm.T).contiguous()
    opB = (B_cm if transb_int == 0 else B_cm.T).contiguous()

    # Step 1: aclnnMuls -> scaled_A = opA * alpha
    scaled_A = torch.empty(m, n, dtype=A.dtype, device=A.device)
    opA_t = create_acl_tensor(opA, list(opA.shape), list(opA.stride()))
    alpha_s = create_acl_scalar(alpha, ACL_FLOAT)
    sA_t = create_acl_tensor(scaled_A, list(scaled_A.shape), list(scaled_A.stride()))
    two_stage_launch('aclnnMulsGetWorkspaceSize', 'aclnnMuls',
                     [opA_t, alpha_s, sA_t])
    destroy_acl_tensor(opA_t)
    destroy_acl_tensor(sA_t)
    destroy_acl_scalar(alpha_s)

    # Step 2: aclnnAdd -> result = scaled_A + beta * opB
    result = torch.empty(m, n, dtype=A.dtype, device=A.device)
    sA_t2 = create_acl_tensor(scaled_A, list(scaled_A.shape), list(scaled_A.stride()))
    opB_t = create_acl_tensor(opB, list(opB.shape), list(opB.stride()))
    beta_s = create_acl_scalar(beta, ACL_FLOAT)
    res_t = create_acl_tensor(result, list(result.shape), list(result.stride()))
    two_stage_launch('aclnnAddGetWorkspaceSize', 'aclnnAdd',
                     [sA_t2, opB_t, beta_s, res_t])
    destroy_acl_tensor(sA_t2)
    destroy_acl_tensor(opB_t)
    destroy_acl_scalar(beta_s)
    destroy_acl_tensor(res_t)

    C_cm.copy_(result)
    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    CUBLAS_OP_N = 0
    m, n = 64, 32
    alpha, beta = 1.25, -0.75

    A = torch.randn(m, n, dtype=torch.float32, device=device)
    B = torch.randn(m, n, dtype=torch.float32, device=device)
    A_ref, B_ref = A.clone(), B.clone()

    A_cm = A.t().contiguous()
    B_cm = B.t().contiguous()
    C_cm = torch.empty(n, m, dtype=torch.float32, device=device)

    out_cm = cublasSgeam(CUBLAS_OP_N, CUBLAS_OP_N, m, n, alpha, A_cm, m, beta, B_cm, m, C_cm, m)
    result = out_cm.t()
    expected = alpha * A_ref + beta * B_ref
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("\n✓ cublasSgeam all tests passed")
