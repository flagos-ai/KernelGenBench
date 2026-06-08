import torch
import ctypes

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_scalar, destroy_acl_scalar,
                           two_stage_launch, ACL_COMPLEX64, map_transpose,
                           map_fill_upper, cuda_dtype_to_torch)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_scalar, destroy_acl_scalar,
        two_stage_launch, ACL_COMPLEX64, map_transpose,
        map_fill_upper, cuda_dtype_to_torch)


def cublasCsyrkEx(uplo, trans, n, k, alpha, A, Atype, lda, beta, C, Ctype, ldc):
    """CANN baseline for cublasCsyrkEx using ctypes aclnn C API.

    Mixed-precision syrk: C = alpha * A * A^T + beta * C (complex64)
    Atype, Ctype are cudaDataType enums (accepted for API compat).
    Data is in column-major layout.
    Uses aclnnMm + aclnnMuls + aclnnAdd.
    """
    if isinstance(trans, str):
        trans_int = 0 if trans.upper() == 'N' else 1
    else:
        trans_int = 1 if trans in (1, 2) else 0

    A_flat = A.reshape(-1)
    C_flat = C.reshape(-1)

    if trans_int == 0:
        A_cm = torch.as_strided(A_flat, (n, k), (1, lda))
    else:
        A_cm = torch.as_strided(A_flat, (k, n), (1, lda))
    C_cm = torch.as_strided(C_flat, (n, n), (1, ldc))

    A_cont = A_cm.contiguous()
    C_cont = C_cm.contiguous()

    if trans_int == 0:
        opA = A_cont
        opB = A_cont.T.contiguous()
    else:
        opA = A_cont.T.contiguous()
        opB = A_cont

    mm_out = torch.empty(n, n, dtype=A.dtype, device=A.device)
    opA_t = create_acl_tensor(opA, list(opA.shape), list(opA.stride()))
    opB_t = create_acl_tensor(opB, list(opB.shape), list(opB.stride()))
    mm_out_t = create_acl_tensor(mm_out, list(mm_out.shape), list(mm_out.stride()))
    two_stage_launch('aclnnMmGetWorkspaceSize', 'aclnnMm',
                     [opA_t, opB_t, mm_out_t, ctypes.c_int8(0)])
    destroy_acl_tensor(opA_t)
    destroy_acl_tensor(opB_t)
    destroy_acl_tensor(mm_out_t)

    scaled = torch.empty_like(mm_out)
    mm_t2 = create_acl_tensor(mm_out, list(mm_out.shape), list(mm_out.stride()))
    alpha_s = create_acl_scalar(complex(alpha), ACL_COMPLEX64)
    scaled_t = create_acl_tensor(scaled, list(scaled.shape), list(scaled.stride()))
    two_stage_launch('aclnnMulsGetWorkspaceSize', 'aclnnMuls',
                     [mm_t2, alpha_s, scaled_t])
    destroy_acl_tensor(mm_t2)
    destroy_acl_tensor(scaled_t)
    destroy_acl_scalar(alpha_s)

    result = torch.empty_like(C_cont)
    sc_t = create_acl_tensor(scaled, list(scaled.shape), list(scaled.stride()))
    c_t = create_acl_tensor(C_cont, list(C_cont.shape), list(C_cont.stride()))
    beta_s = create_acl_scalar(complex(beta), ACL_COMPLEX64)
    res_t = create_acl_tensor(result, list(result.shape), list(result.stride()))
    two_stage_launch('aclnnAddGetWorkspaceSize', 'aclnnAdd',
                     [sc_t, c_t, beta_s, res_t])
    destroy_acl_tensor(sc_t)
    destroy_acl_tensor(c_t)
    destroy_acl_scalar(beta_s)
    destroy_acl_tensor(res_t)

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
    CUBLAS_OP_N = 0
    CUDA_C_32F = 8

    n, k = 3, 2
    alpha = complex(1.0, 0.5)
    beta = complex(0.0, 0.0)

    A = torch.randn(n, k, dtype=torch.complex64, device=device)
    C = torch.zeros(n, n, dtype=torch.complex64, device=device)
    A_orig, C_orig = A.clone(), C.clone()

    A_cm = A.t().contiguous()
    C_cm = C.t().contiguous()

    result_cm = cublasCsyrkEx(CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, k, alpha, A_cm, CUDA_C_32F, n, beta, C_cm, CUDA_C_32F, n)
    result = result_cm.t().contiguous()

    expected_full = alpha * (A_orig @ A_orig.t()) + beta * C_orig
    expected = torch.triu(expected_full)

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("\n✓ cublasCsyrkEx all tests passed")
