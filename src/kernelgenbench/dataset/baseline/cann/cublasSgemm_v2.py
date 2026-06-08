import torch
import ctypes

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_scalar, destroy_acl_scalar,
                           two_stage_launch, ACL_FLOAT)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_scalar, destroy_acl_scalar,
        two_stage_launch, ACL_FLOAT)


def cublasSgemm_v2(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """CANN baseline for cublasSgemm_v2 using ctypes aclnnGemm C API.

    Computes: C = alpha * op(A) @ op(B) + beta * C (float32)
    Data is in column-major layout.
    """
    # Parse transpose flags
    if isinstance(transa, str):
        transa_int = 0 if transa.upper() == 'N' else 1
    else:
        transa_int = 1 if transa in (1, 2) else 0
    if isinstance(transb, str):
        transb_int = 0 if transb.upper() == 'N' else 1
    else:
        transb_int = 1 if transb in (1, 2) else 0

    # Interpret flat memory as column-major matrices
    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)
    C_flat = C.reshape(-1)

    # Column-major views with as_strided
    A_cm = torch.as_strided(A_flat, (m, k) if transa_int == 0 else (k, m), (1, lda))
    B_cm = torch.as_strided(B_flat, (k, n) if transb_int == 0 else (n, k), (1, ldb))
    C_cm = torch.as_strided(C_flat, (m, n), (1, ldc))

    # Make contiguous for aclnn
    A_cont = A_cm.contiguous()
    B_cont = B_cm.contiguous()
    C_cont = C_cm.contiguous()

    # Create aclTensors
    A_shape = list(A_cont.shape)
    B_shape = list(B_cont.shape)
    C_shape = list(C_cont.shape)

    A_stride = [A_cont.stride(0), A_cont.stride(1)]
    B_stride = [B_cont.stride(0), B_cont.stride(1)]
    C_stride = [C_cont.stride(0), C_cont.stride(1)]

    A_t = create_acl_tensor(A_cont, A_shape, A_stride)
    B_t = create_acl_tensor(B_cont, B_shape, B_stride)
    C_t = create_acl_tensor(C_cont, C_shape, C_stride)
    out_t = create_acl_tensor(C_cont, C_shape, C_stride)

    # Create scalars
    alpha_s = create_acl_scalar(alpha, ACL_FLOAT)
    beta_s = create_acl_scalar(beta, ACL_FLOAT)

    # aclnnGemm(self, mat2, mat3, alpha, beta, transA, transB, out, cubeMathType)
    two_stage_launch('aclnnGemmGetWorkspaceSize', 'aclnnGemm',
                     [A_t, B_t, C_t, alpha_s, beta_s,
                      ctypes.c_int64(transa_int), ctypes.c_int64(transb_int),
                      out_t, ctypes.c_int8(0)])

    # Cleanup
    destroy_acl_tensor(A_t)
    destroy_acl_tensor(B_t)
    destroy_acl_tensor(C_t)
    destroy_acl_tensor(out_t)
    destroy_acl_scalar(alpha_s)
    destroy_acl_scalar(beta_s)

    # Copy result back
    C_cm.copy_(C_cont)
    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    m, n, k = 64, 48, 32
    alpha, beta = 1.5, 0.75

    A = torch.randn(m, k, dtype=torch.float32, device=device)
    B = torch.randn(k, n, dtype=torch.float32, device=device)
    C = torch.randn(m, n, dtype=torch.float32, device=device)

    A0, B0, C0 = A.clone(), B.clone(), C.clone()

    # Convert to column-major
    A_cm = A.t().contiguous()
    B_cm = B.t().contiguous()
    C_cm = C.t().contiguous()

    lda, ldb, ldc = m, k, m

    result = cublasSgemm_v2('N', 'N', m, n, k, alpha, A_cm, lda, B_cm, ldb, beta, C_cm, ldc)
    result_rm = result.t()

    expected = alpha * (A0 @ B0) + beta * C0
    torch.testing.assert_close(result_rm, expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasSgemm_v2 test passed")
