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


def cublasSgemmBatched_64(transa, transb, m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount):
    """CANN baseline for cublasSgemmBatched_64 using ctypes aclnn C API (float32).

    Pointer-array batched gemm: C[i] = alpha * op(A[i]) @ op(B[i]) + beta * C[i]
    Aarray/Barray/Carray are int64 tensors holding device pointers.
    The actual matrix data is accessed via those pointers.
    We iterate per-batch using aclnnMm + aclnnMuls + aclnnAdd.
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
    dtype = torch.float32
    elem_size = 4  # float32
    A_ptrs = Aarray.cpu().tolist()
    B_ptrs = Barray.cpu().tolist()
    C_ptrs = Carray.cpu().tolist()

    a_rows = m if transa_int == 0 else k
    a_cols = k if transa_int == 0 else m
    b_rows = k if transb_int == 0 else n
    b_cols = n if transb_int == 0 else k

    for i in range(batchCount):
        # Create column-major views from device pointers
        # The pointers point to existing tensor storage, so we create
        # tensors that share the same memory using from_blob-style access
        a_numel = lda * a_cols
        b_numel = ldb * b_cols
        c_numel = ldc * n

        # Create wrapper tensors from device pointers
        a_base = torch.empty(a_numel, dtype=dtype, device=device)
        b_base = torch.empty(b_numel, dtype=dtype, device=device)
        c_base = torch.empty(c_numel, dtype=dtype, device=device)

        # On CPU, copy data from pointer locations
        if device.type == 'cpu':
            ctypes.memmove(a_base.data_ptr(), int(A_ptrs[i]), a_numel * elem_size)
            ctypes.memmove(b_base.data_ptr(), int(B_ptrs[i]), b_numel * elem_size)
            ctypes.memmove(c_base.data_ptr(), int(C_ptrs[i]), c_numel * elem_size)
        else:
            # On NPU, the pointers are device pointers to existing tensors
            # We use the acl runtime to copy, but since the data is already
            # on device and referenced by the pointer array, we create
            # tensors that view the same storage
            a_base = torch.empty(a_numel, dtype=dtype, device=device)
            b_base = torch.empty(b_numel, dtype=dtype, device=device)
            c_base = torch.empty(c_numel, dtype=dtype, device=device)
            # Copy device-to-device
            from ._backend import get_acl_lib
            lib = get_acl_lib()
            lib.aclrtMemcpy(ctypes.c_void_p(a_base.data_ptr()),
                           ctypes.c_size_t(a_numel * elem_size),
                           ctypes.c_void_p(int(A_ptrs[i])),
                           ctypes.c_size_t(a_numel * elem_size),
                           ctypes.c_int(4))  # ACL_MEMCPY_DEVICE_TO_DEVICE
            lib.aclrtMemcpy(ctypes.c_void_p(b_base.data_ptr()),
                           ctypes.c_size_t(b_numel * elem_size),
                           ctypes.c_void_p(int(B_ptrs[i])),
                           ctypes.c_size_t(b_numel * elem_size),
                           ctypes.c_int(4))
            lib.aclrtMemcpy(ctypes.c_void_p(c_base.data_ptr()),
                           ctypes.c_size_t(c_numel * elem_size),
                           ctypes.c_void_p(int(C_ptrs[i])),
                           ctypes.c_size_t(c_numel * elem_size),
                           ctypes.c_int(4))

        A_cm = torch.as_strided(a_base, (a_rows, a_cols), (1, lda))
        B_cm = torch.as_strided(b_base, (b_rows, b_cols), (1, ldb))
        C_cm = torch.as_strided(c_base, (m, n), (1, ldc))

        opA = (A_cm if transa_int == 0 else A_cm.T).contiguous()
        opB = (B_cm if transb_int == 0 else B_cm.T).contiguous()
        C_cont = C_cm.contiguous()

        # aclnnMm: mm_out = opA @ opB
        mm_out = torch.empty(m, n, dtype=dtype, device=device)
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

        # Write result back to column-major C and copy to original pointer
        C_cm.copy_(result)
        if device.type == 'cpu':
            ctypes.memmove(int(C_ptrs[i]), c_base.data_ptr(), c_numel * elem_size)
        else:
            lib.aclrtMemcpy(ctypes.c_void_p(int(C_ptrs[i])),
                           ctypes.c_size_t(c_numel * elem_size),
                           ctypes.c_void_p(c_base.data_ptr()),
                           ctypes.c_size_t(c_numel * elem_size),
                           ctypes.c_int(4))

    return Carray


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    m, n, k = 4, 5, 6
    batchCount = 3
    alpha, beta = 1.5, -0.25

    A_list = [torch.randn(m, k, dtype=torch.float32, device=device) for _ in range(batchCount)]
    B_list = [torch.randn(k, n, dtype=torch.float32, device=device) for _ in range(batchCount)]
    C_list = [torch.randn(m, n, dtype=torch.float32, device=device) for _ in range(batchCount)]
    A_ref = [a.clone() for a in A_list]
    B_ref = [b.clone() for b in B_list]
    C_ref = [c.clone() for c in C_list]

    A_cm = [a.t().contiguous() for a in A_list]
    B_cm = [b.t().contiguous() for b in B_list]
    C_cm = [c.t().contiguous() for c in C_list]

    Aarray = torch.tensor([int(x.data_ptr()) for x in A_cm], dtype=torch.int64, device=device)
    Barray = torch.tensor([int(x.data_ptr()) for x in B_cm], dtype=torch.int64, device=device)
    Carray = torch.tensor([int(x.data_ptr()) for x in C_cm], dtype=torch.int64, device=device)

    lda, ldb, ldc = m, k, m

    out = cublasSgemmBatched_64('N', 'N', m, n, k, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, batchCount)

    for i in range(batchCount):
        expected = alpha * (A_ref[i] @ B_ref[i]) + beta * C_ref[i]
        result_rm = C_cm[i].t().contiguous()
        torch.testing.assert_close(result_rm, expected, rtol=1e-2, atol=1e-2)

    print("\n✓ cublasSgemmBatched_64 all tests passed")
