import torch
import ctypes

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           two_stage_launch,
                           map_fill_upper, map_unit_diagonal, map_transpose)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        two_stage_launch,
        map_fill_upper, map_unit_diagonal, map_transpose)


def cublasStrsv_v2(uplo, trans, diag, n, A, lda, x, incx):
    """CANN baseline for cublasStrsv_v2 using ctypes aclnn C API.

    Solves op(A) * x = b where A is triangular. Result overwrites x. (float32)
    Data is in column-major layout.
    Uses aclnnTriangularSolve (reshape x to [n, 1]).
    """
    is_upper = map_fill_upper(uplo)
    is_trans = bool(map_transpose(trans))
    is_unit = map_unit_diagonal(diag)

    A_flat = A.reshape(-1)
    A_cm = torch.as_strided(A_flat, (n, n), (1, lda))
    A_cont = A_cm.contiguous()

    # Extract strided x as column vector
    x_vec = x[::incx][:n].contiguous().reshape(n, 1)

    out = torch.empty(n, 1, dtype=A.dtype, device=A.device)
    clonedA = torch.empty_like(A_cont)

    rhs_t = create_acl_tensor(x_vec, list(x_vec.shape), list(x_vec.stride()))
    tri_t = create_acl_tensor(A_cont, list(A_cont.shape), list(A_cont.stride()))
    out_t = create_acl_tensor(out, list(out.shape), list(out.stride()))
    clA_t = create_acl_tensor(clonedA, list(clonedA.shape), list(clonedA.stride()))

    two_stage_launch('aclnnTriangularSolveGetWorkspaceSize', 'aclnnTriangularSolve',
                     [rhs_t, tri_t,
                      ctypes.c_bool(is_upper),
                      ctypes.c_bool(is_trans),
                      ctypes.c_bool(is_unit),
                      out_t, clA_t])

    destroy_acl_tensor(rhs_t)
    destroy_acl_tensor(tri_t)
    destroy_acl_tensor(out_t)
    destroy_acl_tensor(clA_t)

    # Write back to strided x
    x[::incx][:n] = out.reshape(-1)
    return x


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    n = 4
    A_full = torch.randn(n, n, dtype=torch.float32, device=device)
    A_upper = torch.triu(A_full) + torch.eye(n, dtype=torch.float32, device=device) * 5.0

    b = torch.randn(n, dtype=torch.float32, device=device)
    x_vec = b.clone()

    uplo_lower = 0
    trans_T = 1
    diag_nonunit = 0
    lda = n
    incx = 1

    result = cublasStrsv_v2(uplo_lower, trans_T, diag_nonunit, n, A_upper, lda, x_vec, incx)

    x_expected = torch.linalg.solve_triangular(A_upper, b.unsqueeze(1), upper=True, left=True, unitriangular=False).squeeze(1)
    torch.testing.assert_close(result, x_expected, rtol=1e-4, atol=1e-4)
    print("\n✓ cublasStrsv_v2 all tests passed")
