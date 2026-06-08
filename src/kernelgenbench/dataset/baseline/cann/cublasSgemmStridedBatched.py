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


def cublasSgemmStridedBatched(transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb, strideB, beta, C, ldc, strideC, batchCount):
    """CANN baseline for cublasSgemmStridedBatched using ctypes aclnn C API.

    Computes C[i] = alpha * op(A[i]) @ op(B[i]) + beta * C[i] (float32)
    Data is in column-major layout with strided batches.
    Uses aclnnBatchMatMul + aclnnMuls + aclnnAdd.
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

    # Build 3D column-major views for each batch
    # For transa==0: each A batch is (m, k) col-major with stride (1, lda)
    # For transa==1: stored as (k, m) col-major, we want (m, k) so transpose
    A_batches = []
    B_batches = []
    C_batches = []
    for i in range(batchCount):
        a_off = i * strideA
        b_off = i * strideB
        c_off = i * strideC
        A_cm = torch.as_strided(A_flat[a_off:], (m, k) if transa_int == 0 else (k, m), (1, lda))
        B_cm = torch.as_strided(B_flat[b_off:], (k, n) if transb_int == 0 else (n, k), (1, ldb))
        C_cm = torch.as_strided(C_flat[c_off:], (m, n), (1, ldc))
        opA = A_cm if transa_int == 0 else A_cm.T
        opB = B_cm if transb_int == 0 else B_cm.T
        A_batches.append(opA.contiguous())
        B_batches.append(opB.contiguous())
        C_batches.append(C_cm.contiguous())

    # Stack into 3D tensors [batch, M, K], [batch, K, N], [batch, M, N]
    A_3d = torch.stack(A_batches, dim=0)
    B_3d = torch.stack(B_batches, dim=0)
    C_3d = torch.stack(C_batches, dim=0)

    # Output for bmm
    out_bmm = torch.empty_like(C_3d)

    # Step 1: BatchMatMul -> out_bmm = A_3d @ B_3d
    A_3t = create_acl_tensor(A_3d, list(A_3d.shape), list(A_3d.stride()))
    B_3t = create_acl_tensor(B_3d, list(B_3d.shape), list(B_3d.stride()))
    out_bmm_t = create_acl_tensor(out_bmm, list(out_bmm.shape), list(out_bmm.stride()))

    two_stage_launch('aclnnBatchMatMulGetWorkspaceSize', 'aclnnBatchMatMul',
                     [A_3t, B_3t, out_bmm_t, ctypes.c_int8(0)])

    destroy_acl_tensor(A_3t)
    destroy_acl_tensor(B_3t)
    destroy_acl_tensor(out_bmm_t)

    # Step 2: Muls -> scaled = out_bmm * alpha
    scaled = torch.empty_like(out_bmm)
    bmm_t2 = create_acl_tensor(out_bmm, list(out_bmm.shape), list(out_bmm.stride()))
    alpha_s = create_acl_scalar(alpha, ACL_FLOAT)
    scaled_t = create_acl_tensor(scaled, list(scaled.shape), list(scaled.stride()))

    two_stage_launch('aclnnMulsGetWorkspaceSize', 'aclnnMuls',
                     [bmm_t2, alpha_s, scaled_t])

    destroy_acl_tensor(bmm_t2)
    destroy_acl_tensor(scaled_t)
    destroy_acl_scalar(alpha_s)

    # Step 3: Add -> result = scaled + beta * C_3d
    result_3d = torch.empty_like(C_3d)
    scaled_t2 = create_acl_tensor(scaled, list(scaled.shape), list(scaled.stride()))
    C_3t = create_acl_tensor(C_3d, list(C_3d.shape), list(C_3d.stride()))
    beta_s = create_acl_scalar(beta, ACL_FLOAT)
    result_t = create_acl_tensor(result_3d, list(result_3d.shape), list(result_3d.stride()))

    two_stage_launch('aclnnAddGetWorkspaceSize', 'aclnnAdd',
                     [scaled_t2, C_3t, beta_s, result_t])

    destroy_acl_tensor(scaled_t2)
    destroy_acl_tensor(C_3t)
    destroy_acl_scalar(beta_s)
    destroy_acl_tensor(result_t)

    # Write results back to column-major C
    for i in range(batchCount):
        c_off = i * strideC
        C_cm = torch.as_strided(C_flat[c_off:], (m, n), (1, ldc))
        C_cm.copy_(result_3d[i])

    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    batchCount = 8
    m, n, k = 64, 48, 32
    alpha, beta = 1.0, 0.5

    A = torch.randn(batchCount, m, k, dtype=torch.float32, device=device)
    B = torch.randn(batchCount, k, n, dtype=torch.float32, device=device)
    C = torch.randn(batchCount, m, n, dtype=torch.float32, device=device)
    A_original, B_original, C_original = A.clone(), B.clone(), C.clone()

    A_t = A.transpose(-2, -1).contiguous()
    B_t = B.transpose(-2, -1).contiguous()
    C_t = C.transpose(-2, -1).contiguous()

    lda, ldb, ldc = m, k, m
    strideA = lda * k
    strideB = ldb * n
    strideC = ldc * n

    result_t = cublasSgemmStridedBatched(0, 0, m, n, k, alpha, A_t, lda, strideA, B_t, ldb, strideB, beta, C_t, ldc, strideC, batchCount)
    result = result_t.transpose(-2, -1).contiguous()
    expected = alpha * torch.bmm(A_original, B_original) + beta * C_original
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-2)
    print("\n✓ cublasSgemmStridedBatched all tests passed")
