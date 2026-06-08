import torch
import ctypes

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_scalar, destroy_acl_scalar,
                           two_stage_launch, ACL_FLOAT16)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_scalar, destroy_acl_scalar,
        two_stage_launch, ACL_FLOAT16)


def cublasHgemmBatched(transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount):
    """CANN baseline for cublasHgemmBatched using ctypes aclnn C API (float16).

    Pointer-array batched gemm: C[i] = alpha * op(A[i]) @ op(B[i]) + beta * C[i]
    Aarray/Barray/Carray are int64 tensors holding device pointers.
    """
    if isinstance(transa, str):
        transa_int = 0 if transa.upper() == 'N' else 1
    else:
        transa_int = 1 if transa in (1, 2) else 0
    if isinstance(transb, str):
        transb_int = 0 if transb.upper() == 'N' else 1
    else:
        transb_int = 1 if transb in (1, 2) else 0

    device = Aarray.device
    dtype = torch.float16
    elem_size = 2
    A_ptrs = Aarray.cpu().tolist()
    B_ptrs = Barray.cpu().tolist()
    C_ptrs = Carray.cpu().tolist()

    a_rows = m if transa_int == 0 else k
    a_cols = k if transa_int == 0 else m
    b_rows = k if transb_int == 0 else n
    b_cols = n if transb_int == 0 else k

    for i in range(batchCount):
        a_numel = lda * a_cols
        b_numel = ldb * b_cols
        c_numel = ldc * n

        a_base = torch.empty(a_numel, dtype=dtype, device=device)
        b_base = torch.empty(b_numel, dtype=dtype, device=device)
        c_base = torch.empty(c_numel, dtype=dtype, device=device)

        if device.type == 'cpu':
            ctypes.memmove(a_base.data_ptr(), int(A_ptrs[i]), a_numel * elem_size)
            ctypes.memmove(b_base.data_ptr(), int(B_ptrs[i]), b_numel * elem_size)
            ctypes.memmove(c_base.data_ptr(), int(C_ptrs[i]), c_numel * elem_size)
        else:
            from ._backend import get_acl_lib
            lib = get_acl_lib()
            for dst, src, sz in [(a_base, A_ptrs[i], a_numel), (b_base, B_ptrs[i], b_numel), (c_base, C_ptrs[i], c_numel)]:
                lib.aclrtMemcpy(ctypes.c_void_p(dst.data_ptr()), ctypes.c_size_t(sz * elem_size),
                               ctypes.c_void_p(int(src)), ctypes.c_size_t(sz * elem_size), ctypes.c_int(4))

        A_cm = torch.as_strided(a_base, (a_rows, a_cols), (1, lda))
        B_cm = torch.as_strided(b_base, (b_rows, b_cols), (1, ldb))
        C_cm = torch.as_strided(c_base, (m, n), (1, ldc))

        opA = (A_cm if transa_int == 0 else A_cm.T).contiguous()
        opB = (B_cm if transb_int == 0 else B_cm.T).contiguous()
        C_cont = C_cm.contiguous()

        mm_out = torch.empty(m, n, dtype=dtype, device=device)
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
        alpha_s = create_acl_scalar(alpha, ACL_FLOAT16)
        scaled_t = create_acl_tensor(scaled, list(scaled.shape), list(scaled.stride()))
        two_stage_launch('aclnnMulsGetWorkspaceSize', 'aclnnMuls',
                         [mm_t2, alpha_s, scaled_t])
        destroy_acl_tensor(mm_t2)
        destroy_acl_tensor(scaled_t)
        destroy_acl_scalar(alpha_s)

        result = torch.empty_like(C_cont)
        sc_t = create_acl_tensor(scaled, list(scaled.shape), list(scaled.stride()))
        c_t = create_acl_tensor(C_cont, list(C_cont.shape), list(C_cont.stride()))
        beta_s = create_acl_scalar(beta, ACL_FLOAT16)
        res_t = create_acl_tensor(result, list(result.shape), list(result.stride()))
        two_stage_launch('aclnnAddGetWorkspaceSize', 'aclnnAdd',
                         [sc_t, c_t, beta_s, res_t])
        destroy_acl_tensor(sc_t)
        destroy_acl_tensor(c_t)
        destroy_acl_scalar(beta_s)
        destroy_acl_tensor(res_t)

        C_cm.copy_(result)
        if device.type == 'cpu':
            ctypes.memmove(int(C_ptrs[i]), c_base.data_ptr(), c_numel * elem_size)
        else:
            lib.aclrtMemcpy(ctypes.c_void_p(int(C_ptrs[i])), ctypes.c_size_t(c_numel * elem_size),
                           ctypes.c_void_p(c_base.data_ptr()), ctypes.c_size_t(c_numel * elem_size), ctypes.c_int(4))

    return Carray


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    m, n, k = 4, 5, 6
    batchCount = 3
    alpha, beta = 1.0, 0.0

    A_list = [torch.randint(-2, 3, (m, k), dtype=torch.int8, device=device).to(torch.float16) for _ in range(batchCount)]
    B_list = [torch.randint(-2, 3, (k, n), dtype=torch.int8, device=device).to(torch.float16) for _ in range(batchCount)]
    C_list = [torch.zeros((m, n), dtype=torch.float16, device=device) for _ in range(batchCount)]

    A_cm = [a.t().contiguous() for a in A_list]
    B_cm = [b.t().contiguous() for b in B_list]
    C_cm = [torch.zeros((n, m), dtype=torch.float16, device=device) for _ in range(batchCount)]

    A_ptrs = torch.tensor([t.data_ptr() for t in A_cm], dtype=torch.int64, device=device)
    B_ptrs = torch.tensor([t.data_ptr() for t in B_cm], dtype=torch.int64, device=device)
    C_ptrs = torch.tensor([t.data_ptr() for t in C_cm], dtype=torch.int64, device=device)

    lda, ldb, ldc = m, k, m

    ret = cublasHgemmBatched('N', 'N', m, n, k, alpha, A_ptrs, lda, B_ptrs, ldb, beta, C_ptrs, ldc, batchCount)

    for i in range(batchCount):
        expected = A_list[i] @ B_list[i]
        result_rm = C_cm[i].t().contiguous()
        torch.testing.assert_close(result_rm, expected, rtol=1e-5, atol=1e-5)

    print("\n✓ cublasHgemmBatched all tests passed")
