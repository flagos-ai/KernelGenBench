import torch
import ctypes

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_scalar, destroy_acl_scalar,
                           two_stage_launch, ACL_COMPLEX128,
                           map_fill_upper, map_side_left,
                           map_unit_diagonal, map_transpose)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_scalar, destroy_acl_scalar,
        two_stage_launch, ACL_COMPLEX128,
        map_fill_upper, map_side_left,
        map_unit_diagonal, map_transpose)


def cublasZtrsmBatched(side, uplo, trans, diag, m, n, alpha, Aarray, lda, Barray, ldb, batchCount):
    """CANN baseline for cublasZtrsmBatched using ctypes aclnn C API (complex128).

    Batched triangular solve: op(A[i]) * X[i] = alpha * B[i] (side=LEFT)
    Aarray/Barray are int64 tensors holding device pointers.
    Uses aclnnTriangularSolve per batch + aclnnMuls for alpha scaling.
    """
    if isinstance(side, str):
        side = 0 if side == 'L' else 1
    if isinstance(uplo, str):
        uplo = 0 if uplo == 'U' else 1
    if isinstance(trans, str):
        trans_val = 0 if trans == 'N' else (1 if trans == 'T' else 2)
    else:
        trans_val = trans
    if isinstance(diag, str):
        diag = 0 if diag == 'N' else 1

    is_left = map_side_left(side)
    is_upper = map_fill_upper(uplo)
    is_trans = bool(map_transpose(trans_val))
    is_unit = map_unit_diagonal(diag)
    is_conj = trans_val == 2

    device = Aarray.device
    dtype = torch.complex128
    elem_size = 16
    A_ptrs = Aarray.cpu().tolist()
    B_ptrs = Barray.cpu().tolist()

    dim_a = m if is_left else n

    for i in range(batchCount):
        a_numel = lda * dim_a
        b_numel = ldb * n

        a_base = torch.empty(a_numel, dtype=dtype, device=device)
        b_base = torch.empty(b_numel, dtype=dtype, device=device)

        if device.type == 'cpu':
            ctypes.memmove(a_base.data_ptr(), int(A_ptrs[i]), a_numel * elem_size)
            ctypes.memmove(b_base.data_ptr(), int(B_ptrs[i]), b_numel * elem_size)
        else:
            from ._backend import get_acl_lib
            lib = get_acl_lib()
            for dst, src, sz in [(a_base, A_ptrs[i], a_numel), (b_base, B_ptrs[i], b_numel)]:
                lib.aclrtMemcpy(ctypes.c_void_p(dst.data_ptr()), ctypes.c_size_t(sz * elem_size),
                               ctypes.c_void_p(int(src)), ctypes.c_size_t(sz * elem_size), ctypes.c_int(4))

        A_cm = torch.as_strided(a_base, (dim_a, dim_a), (1, lda)).contiguous()
        B_cm = torch.as_strided(b_base, (m, n), (1, ldb)).contiguous()

        if is_conj:
            A_cm = A_cm.conj().resolve_conj().contiguous()

        # Scale B by alpha
        if alpha != 1.0:
            scaled_B = torch.empty_like(B_cm)
            B_t = create_acl_tensor(B_cm, list(B_cm.shape), list(B_cm.stride()))
            alpha_s = create_acl_scalar(complex(alpha), ACL_COMPLEX128)
            sB_t = create_acl_tensor(scaled_B, list(scaled_B.shape), list(scaled_B.stride()))
            two_stage_launch('aclnnMulsGetWorkspaceSize', 'aclnnMuls',
                             [B_t, alpha_s, sB_t])
            destroy_acl_tensor(B_t)
            destroy_acl_scalar(alpha_s)
            destroy_acl_tensor(sB_t)
            B_cm = scaled_B

        if is_left:
            rhs = B_cm
            tri = A_cm
            upper_flag = is_upper
            trans_flag = is_trans
        else:
            rhs = B_cm.T.contiguous()
            tri = A_cm.T.contiguous()
            upper_flag = not is_upper
            trans_flag = not is_trans

        out = torch.empty_like(rhs)
        clonedA = torch.empty_like(tri)

        rhs_t = create_acl_tensor(rhs, list(rhs.shape), list(rhs.stride()))
        tri_t = create_acl_tensor(tri, list(tri.shape), list(tri.stride()))
        out_t = create_acl_tensor(out, list(out.shape), list(out.stride()))
        clA_t = create_acl_tensor(clonedA, list(clonedA.shape), list(clonedA.stride()))

        two_stage_launch('aclnnTriangularSolveGetWorkspaceSize', 'aclnnTriangularSolve',
                         [rhs_t, tri_t,
                          ctypes.c_bool(upper_flag),
                          ctypes.c_bool(trans_flag),
                          ctypes.c_bool(is_unit),
                          out_t, clA_t])

        destroy_acl_tensor(rhs_t)
        destroy_acl_tensor(tri_t)
        destroy_acl_tensor(out_t)
        destroy_acl_tensor(clA_t)

        if not is_left:
            out = out.T.contiguous()

        B_cm_out = torch.as_strided(b_base, (m, n), (1, ldb))
        B_cm_out.copy_(out)

        if device.type == 'cpu':
            ctypes.memmove(int(B_ptrs[i]), b_base.data_ptr(), b_numel * elem_size)
        else:
            lib.aclrtMemcpy(ctypes.c_void_p(int(B_ptrs[i])), ctypes.c_size_t(b_numel * elem_size),
                           ctypes.c_void_p(b_base.data_ptr()), ctypes.c_size_t(b_numel * elem_size), ctypes.c_int(4))

    return Barray


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'
    dtype = torch.complex128

    batchCount = 3
    m, n = 3, 3
    alpha = 1.2 - 0.7j

    A_list, B_list = [], []
    for i in range(batchCount):
        A = torch.randn((m, m), dtype=dtype, device=device)
        A = torch.triu(A) + torch.diag(torch.ones(m, dtype=dtype, device=device) * (2.0 + 0.0j))
        B = torch.randn((m, n), dtype=dtype, device=device)
        A_list.append(A)
        B_list.append(B)

    A_list_ref = [A.clone() for A in A_list]
    B_list_ref = [B.clone() for B in B_list]

    A_ptrs = torch.tensor([A.data_ptr() for A in A_list], dtype=torch.int64, device=device)
    B_ptrs = torch.tensor([B.data_ptr() for B in B_list], dtype=torch.int64, device=device)

    result_ptrs = cublasZtrsmBatched('L', 'L', 'N', 'N', m, n, alpha, A_ptrs, m, B_ptrs, m, batchCount)

    for i in range(batchCount):
        A_rm = A_list_ref[i]
        B_rm = B_list_ref[i]
        A_cm_ref = A_rm.t().contiguous()
        B_cm_ref = B_rm.t().contiguous()
        RHS = alpha * B_cm_ref
        X_cm = torch.linalg.solve(A_cm_ref, RHS)
        expected = X_cm.t().contiguous()
        torch.testing.assert_close(B_list[i], expected, rtol=1e-10, atol=1e-10)

    print("\n✓ cublasZtrsmBatched all tests passed")
