import torch
import ctypes

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           two_stage_launch, map_side_left)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        two_stage_launch, map_side_left)


def cublasSdgmm(mode, m, n, A, lda, x, incx, C, ldc):
    """CANN baseline for cublasSdgmm using ctypes aclnn C API.

    C = A * diag(x) (right) or C = diag(x) * A (left) (float32)
    Data is in column-major layout.
    Uses aclnnMul with broadcast.
    mode: 0=LEFT (scale rows), 1=RIGHT (scale columns)
    """
    A_flat = A.reshape(-1)
    C_flat = C.reshape(-1)

    # Column-major views
    A_cm = torch.as_strided(A_flat, (m, n), (1, lda))
    C_cm = torch.as_strided(C_flat, (m, n), (1, ldc))

    # Extract strided x
    x_vec = x[::incx].contiguous()

    # mode=0 (LEFT): scale each row i by x[i] -> broadcast x as (m, 1)
    # mode=1 (RIGHT): scale each column j by x[j] -> broadcast x as (1, n)
    A_cont = A_cm.contiguous()
    if mode == 0:  # LEFT
        x_bc = x_vec[:m].reshape(m, 1).contiguous()
    else:  # RIGHT
        x_bc = x_vec[:n].reshape(1, n).contiguous()

    result = torch.empty(m, n, dtype=A.dtype, device=A.device)

    A_t = create_acl_tensor(A_cont, list(A_cont.shape), list(A_cont.stride()))
    x_t = create_acl_tensor(x_bc, list(x_bc.shape), list(x_bc.stride()))
    res_t = create_acl_tensor(result, list(result.shape), list(result.stride()))

    two_stage_launch('aclnnMulGetWorkspaceSize', 'aclnnMul',
                     [A_t, x_t, res_t])

    destroy_acl_tensor(A_t)
    destroy_acl_tensor(x_t)
    destroy_acl_tensor(res_t)

    C_cm.copy_(result)
    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    CUBLAS_SIDE_LEFT = 0
    CUBLAS_SIDE_RIGHT = 1

    m_orig, n_orig = 4, 6
    A = torch.randn(m_orig, n_orig, dtype=torch.float32, device=device).contiguous()
    A_clone = A.clone()

    # Right multiplication: C = A * diag(x_right)
    x_right = torch.randn(n_orig, dtype=torch.float32, device=device)
    x_right_clone = x_right.clone()
    C_right = torch.empty_like(A)

    out_right = cublasSdgmm(CUBLAS_SIDE_LEFT, n_orig, m_orig, A, n_orig, x_right, 1, C_right, n_orig)
    expected_right = A_clone * x_right_clone.view(1, -1)
    torch.testing.assert_close(out_right, expected_right, rtol=1e-5, atol=1e-5)

    # Left multiplication: C = diag(x_left) * A
    x_left = torch.randn(m_orig, dtype=torch.float32, device=device)
    x_left_clone = x_left.clone()
    C_left = torch.empty_like(A)

    out_left = cublasSdgmm(CUBLAS_SIDE_RIGHT, n_orig, m_orig, A, n_orig, x_left, 1, C_left, n_orig)
    expected_left = x_left_clone.view(-1, 1) * A_clone
    torch.testing.assert_close(out_left, expected_left, rtol=1e-5, atol=1e-5)

    print("\n✓ cublasSdgmm all tests passed")
