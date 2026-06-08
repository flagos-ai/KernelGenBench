import torch
import ctypes

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_scalar, destroy_acl_scalar,
                           two_stage_launch, ACL_FLOAT,
                           map_fill_upper, map_side_left,
                           map_unit_diagonal, map_transpose)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_scalar, destroy_acl_scalar,
        two_stage_launch, ACL_FLOAT,
        map_fill_upper, map_side_left,
        map_unit_diagonal, map_transpose)


def cublasStrsm_v2(side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb):
    """CANN baseline for cublasStrsm_v2 using ctypes aclnn C API.

    Solves op(A) * X = alpha * B (side=LEFT) or X * op(A) = alpha * B (side=RIGHT)
    where A is triangular. Result overwrites B. (float32)
    Data is in column-major layout.
    Uses aclnnTriangularSolve.
    """
    is_left = map_side_left(side)
    is_upper = map_fill_upper(uplo)
    is_trans = bool(map_transpose(trans))
    is_unit = map_unit_diagonal(diag)

    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)

    # Column-major views
    dim_a = m if is_left else n
    A_cm = torch.as_strided(A_flat, (dim_a, dim_a), (1, lda))
    B_cm = torch.as_strided(B_flat, (m, n), (1, ldb))

    A_cont = A_cm.contiguous()
    B_cont = B_cm.contiguous()

    # Scale B by alpha
    if alpha != 1.0:
        scaled_B = torch.empty_like(B_cont)
        B_t = create_acl_tensor(B_cont, list(B_cont.shape), list(B_cont.stride()))
        alpha_s = create_acl_scalar(alpha, ACL_FLOAT)
        sB_t = create_acl_tensor(scaled_B, list(scaled_B.shape), list(scaled_B.stride()))
        two_stage_launch('aclnnMulsGetWorkspaceSize', 'aclnnMuls',
                         [B_t, alpha_s, sB_t])
        destroy_acl_tensor(B_t)
        destroy_acl_scalar(alpha_s)
        destroy_acl_tensor(sB_t)
        B_cont = scaled_B

    # aclnnTriangularSolve(self, A, upper, transpose, unitriangular, out, clonedA)
    # self = B (RHS), A = triangular matrix
    # For side=RIGHT: X * A = B => A^T * X^T = B^T, solve as left-side
    if is_left:
        rhs = B_cont
        tri = A_cont
        upper_flag = is_upper
        trans_flag = is_trans
    else:
        # Transform to left-side: A^T * X^T = B^T
        rhs = B_cont.T.contiguous()
        tri = A_cont.T.contiguous()
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

    B_cm.copy_(out)
    return B


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    CUBLAS_SIDE_LEFT = 0
    CUBLAS_FILL_MODE_LOWER = 0
    CUBLAS_OP_N = 0
    CUBLAS_DIAG_NON_UNIT = 0

    m, n = 4, 1
    alpha = 0.5

    A_full = torch.randn(m, m, dtype=torch.float32, device=device)
    A = torch.triu(A_full) + torch.eye(m, dtype=torch.float32, device=device) * 3.0
    B = torch.randn(m, n, dtype=torch.float32, device=device)
    A_in, B_in = A.clone(), B.clone()

    result = cublasStrsm_v2(CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                            m, n, alpha, A_in, m, B_in, m)

    RHS_T = alpha * B.t()
    Z = torch.linalg.solve_triangular(A.t(), RHS_T.t(), upper=False, left=True, unitriangular=False)
    BT_expected = Z.t()
    torch.testing.assert_close(result.t(), BT_expected, rtol=1e-5, atol=1e-5)
    print("\n✓ cublasStrsm_v2 all tests passed")
