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


def cublasCgemmStridedBatched_64(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    """CANN baseline for cublasCgemmStridedBatched_64 using ctypes aclnn C API (complex64, int64 indices).

    Computes C[i] = alpha * op(A[i]) @ op(B[i]) + beta * C[i]
    Data is in column-major layout with strided batches.
    """
    if isinstance(transa, str):
        transa_int = {'N': 0, 'T': 1, 'C': 2}.get(transa.upper(), 0)
    else:
        transa_int = transa
    if isinstance(transb, str):
        transb_int = {'N': 0, 'T': 1, 'C': 2}.get(transb.upper(), 0)
    else:
        transb_int = transb

    trans_a = 1 if transa_int in (1, 2) else 0
    trans_b = 1 if transb_int in (1, 2) else 0

    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)
    C_flat = C.reshape(-1)

    A_batches, B_batches, C_batches = [], [], []
    for i in range(batchCount):
        A_cm = torch.as_strided(A_flat[i * strideA:], (m, k) if trans_a == 0 else (k, m), (1, lda))
        B_cm = torch.as_strided(B_flat[i * strideB:], (k, n) if trans_b == 0 else (n, k), (1, ldb))
        C_cm = torch.as_strided(C_flat[i * strideC:], (m, n), (1, ldc))
        opA = A_cm if trans_a == 0 else A_cm.T
        opB = B_cm if trans_b == 0 else B_cm.T
        if transa_int == 2:
            opA = opA.conj().resolve_conj()
        if transb_int == 2:
            opB = opB.conj().resolve_conj()
        A_batches.append(opA.contiguous())
        B_batches.append(opB.contiguous())
        C_batches.append(C_cm.contiguous())

    A_3d = torch.stack(A_batches, dim=0)
    B_3d = torch.stack(B_batches, dim=0)
    C_3d = torch.stack(C_batches, dim=0)
    out_bmm = torch.empty_like(C_3d)

    A_3t = create_acl_tensor(A_3d, list(A_3d.shape), list(A_3d.stride()))
    B_3t = create_acl_tensor(B_3d, list(B_3d.shape), list(B_3d.stride()))
    out_bmm_t = create_acl_tensor(out_bmm, list(out_bmm.shape), list(out_bmm.stride()))
    two_stage_launch('aclnnBatchMatMulGetWorkspaceSize', 'aclnnBatchMatMul',
                     [A_3t, B_3t, out_bmm_t, ctypes.c_int8(0)])
    destroy_acl_tensor(A_3t)
    destroy_acl_tensor(B_3t)
    destroy_acl_tensor(out_bmm_t)

    scaled = torch.empty_like(out_bmm)
    bmm_t2 = create_acl_tensor(out_bmm, list(out_bmm.shape), list(out_bmm.stride()))
    alpha_s = create_acl_scalar(complex(alpha), ACL_COMPLEX64)
    scaled_t = create_acl_tensor(scaled, list(scaled.shape), list(scaled.stride()))
    two_stage_launch('aclnnMulsGetWorkspaceSize', 'aclnnMuls',
                     [bmm_t2, alpha_s, scaled_t])
    destroy_acl_tensor(bmm_t2)
    destroy_acl_tensor(scaled_t)
    destroy_acl_scalar(alpha_s)

    result_3d = torch.empty_like(C_3d)
    scaled_t2 = create_acl_tensor(scaled, list(scaled.shape), list(scaled.stride()))
    C_3t = create_acl_tensor(C_3d, list(C_3d.shape), list(C_3d.stride()))
    beta_s = create_acl_scalar(complex(beta), ACL_COMPLEX64)
    result_t = create_acl_tensor(result_3d, list(result_3d.shape), list(result_3d.stride()))
    two_stage_launch('aclnnAddGetWorkspaceSize', 'aclnnAdd',
                     [scaled_t2, C_3t, beta_s, result_t])
    destroy_acl_tensor(scaled_t2)
    destroy_acl_tensor(C_3t)
    destroy_acl_scalar(beta_s)
    destroy_acl_tensor(result_t)

    for i in range(batchCount):
        C_cm = torch.as_strided(C_flat[i * strideC:], (m, n), (1, ldc))
        C_cm.copy_(result_3d[i])
    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    batchCount = 2
    m, n, k = 4, 5, 3
    alpha = 1.3 + 0.2j
    beta = 0.7 - 0.1j

    A_rm = torch.randn(batchCount, m, k, dtype=torch.complex64, device='cpu').to(device)
    B_rm = torch.randn(batchCount, k, n, dtype=torch.complex64, device='cpu').to(device)
    C_rm = torch.randn(batchCount, m, n, dtype=torch.complex64, device='cpu').to(device)
    A_ref, B_ref, C_ref = A_rm.clone(), B_rm.clone(), C_rm.clone()

    lda, ldb, ldc = n, k, n
    strideA, strideB, strideC = k * n, m * k, m * n

    C_out = cublasCgemmStridedBatched_64('N', 'N', n, m, k, alpha, B_rm, lda, strideA, A_rm, ldb, strideB, beta, C_rm, ldc, strideC, batchCount)
    expected = alpha * torch.matmul(A_ref, B_ref) + beta * C_ref
    torch.testing.assert_close(C_rm.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
    print("\n✓ cublasCgemmStridedBatched_64 all tests passed")
