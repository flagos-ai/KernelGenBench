import torch
import ctypes

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_scalar, destroy_acl_scalar,
                           two_stage_launch, ACL_FLOAT, map_transpose,
                           map_fill_upper)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_scalar, destroy_acl_scalar,
        two_stage_launch, ACL_FLOAT, map_transpose,
        map_fill_upper)


def cublasSsyrk_v2(uplo, trans, n, k, alpha, A, lda, beta, C, ldc):
    """CANN baseline for cublasSsyrk_v2 using ctypes aclnn C API.

    Computes C = alpha * A * A^T + beta * C (float32)
    (when trans=N) or C = alpha * A^T * A + beta * C (when trans=T)
    Data is in column-major layout.
    Uses aclnnMm + aclnnMuls + aclnnAdd.
    """
    if isinstance(trans, str):
        trans_int = 0 if trans.upper() == 'N' else 1
    else:
        trans_int = 1 if trans in (1, 2) else 0

    A_flat = A.reshape(-1)
    C_flat = C.reshape(-1)

    # Column-major views
    if trans_int == 0:
        A_cm = torch.as_strided(A_flat, (n, k), (1, lda))
    else:
        A_cm = torch.as_strided(A_flat, (k, n), (1, lda))
    C_cm = torch.as_strided(C_flat, (n, n), (1, ldc))

    A_cont = A_cm.contiguous()
    C_cont = C_cm.contiguous()

    # Compute A @ A^T or A^T @ A
    if trans_int == 0:
        # C = alpha * A * A^T + beta * C
        opA = A_cont  # (n, k)
        opB = A_cont.T.contiguous()  # (k, n)
    else:
        # C = alpha * A^T * A + beta * C
        opA = A_cont.T.contiguous()  # (n, k)
        opB = A_cont  # (k, n)

    # aclnnMm: mm_out = opA @ opB
    mm_out = torch.empty(n, n, dtype=A.dtype, device=A.device)
    opA_t = create_acl_tensor(opA, list(opA.shape), list(opA.stride()))
    opB_t = create_acl_tensor(opB, list(opB.shape), list(opB.stride()))
    mm_out_t = create_acl_tensor(mm_out, list(mm_out.shape), list(mm_out.stride()))
    two_stage_launch('aclnnMmGetWorkspaceSize', 'aclnnMm',
                     [opA_t, opB_t, mm_out_t, ctypes.c_int8(0)])
    destroy_acl_tensor(opA_t)
    destroy_acl_tensor(opB_t)
    destroy_acl_tensor(mm_out_t)

    # aclnnMuls: scaled = mm_out * alpha
    scaled = torch.empty_like(mm_out)
    mm_t2 = create_acl_tensor(mm_out, list(mm_out.shape), list(mm_out.stride()))
    alpha_s = create_acl_scalar(alpha, ACL_FLOAT)
    scaled_t = create_acl_tensor(scaled, list(scaled.shape), list(scaled.stride()))
    two_stage_launch('aclnnMulsGetWorkspaceSize', 'aclnnMuls',
                     [mm_t2, alpha_s, scaled_t])
    destroy_acl_tensor(mm_t2)
    destroy_acl_tensor(scaled_t)
    destroy_acl_scalar(alpha_s)

    # aclnnAdd: result = scaled + beta * C_cont
    result = torch.empty_like(C_cont)
    sc_t = create_acl_tensor(scaled, list(scaled.shape), list(scaled.stride()))
    c_t = create_acl_tensor(C_cont, list(C_cont.shape), list(C_cont.stride()))
    beta_s = create_acl_scalar(beta, ACL_FLOAT)
    res_t = create_acl_tensor(result, list(result.shape), list(result.stride()))
    two_stage_launch('aclnnAddGetWorkspaceSize', 'aclnnAdd',
                     [sc_t, c_t, beta_s, res_t])
    destroy_acl_tensor(sc_t)
    destroy_acl_tensor(c_t)
    destroy_acl_scalar(beta_s)
    destroy_acl_tensor(res_t)

    # Only update the requested triangle
    is_upper = map_fill_upper(uplo)
    if is_upper:
        mask = torch.ones(n, n, dtype=torch.bool, device=A.device).triu()
    else:
        mask = torch.ones(n, n, dtype=torch.bool, device=A.device).tril()

    C_cont_new = C_cont.clone()
    C_cont_new[mask] = result[mask]
    C_cm.copy_(C_cont_new)
    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    CUBLAS_FILL_MODE_LOWER = 0
    CUBLAS_OP_T = 1

    n, k = 5, 3
    alpha, beta = 0.75, 0.25

    A = torch.randn(n, k, dtype=torch.float32, device=device).contiguous()
    C = torch.randn(n, n, dtype=torch.float32, device=device).contiguous()
    C_orig = C.clone()

    uplo = CUBLAS_FILL_MODE_LOWER
    trans = CUBLAS_OP_T
    lda, ldc = k, n

    result = cublasSsyrk_v2(uplo, trans, n, k, alpha, A, lda, beta, C, ldc)

    S_full = alpha * (A @ A.t()) + beta * C_orig
    expected = C_orig.clone()
    upper_mask = torch.ones(n, n, dtype=torch.bool, device=device).triu()
    expected[upper_mask] = S_full[upper_mask]

    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)
    print("\n✓ cublasSsyrk_v2 all tests passed")
