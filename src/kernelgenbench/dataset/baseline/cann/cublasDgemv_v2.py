import ctypes
import torch

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_scalar, destroy_acl_scalar,
                           two_stage_launch, torch_dtype_to_acl,
                           map_transpose, ACL_DOUBLE)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_scalar, destroy_acl_scalar,
        two_stage_launch, torch_dtype_to_acl,
        map_transpose, ACL_DOUBLE)


def cublasDgemv_v2(trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    """CANN baseline for cublasDgemv_v2 — ctypes aclnn C API.
    y = alpha * op(A) * x + beta * y  (float64)

    Uses aclnnGemm by reshaping x to a column matrix and y to a column matrix.
    Column-major A is viewed via as_strided with stride (1, lda).
    """
    do_trans = map_transpose(trans)

    # op(A) dimensions: if no transpose, op(A) is (m, n); if transpose, op(A) is (n, m)
    if do_trans:
        rows, cols = n, m
    else:
        rows, cols = m, n

    # Build column-major view of A as (m, n) with stride (1, lda)
    A_flat = A.reshape(-1)
    A_cm = torch.as_strided(A_flat, [m, n], [1, lda])

    # Extract strided vectors
    x_vec = x[::incx][:cols].contiguous()
    y_vec = y[::incy][:rows].contiguous()

    # Reshape for Gemm: x -> (cols, 1), y -> (rows, 1)
    x_mat = x_vec.reshape(cols, 1).contiguous()
    y_mat = y_vec.reshape(rows, 1).contiguous()
    out_mat = torch.empty(rows, 1, dtype=torch.float64, device=A.device)

    # Create ACL tensors
    # aclnnGemm(self, mat2, mat3, alpha, beta, transA, transB, out, cubeMathType)
    # self=A_cm(m,n), mat2=x_mat(cols,1), mat3=y_mat(rows,1)
    # transA=do_trans, transB=0
    # Result: op(A) @ x_mat + beta*y_mat -> out_mat (rows, 1)
    A_t = create_acl_tensor(A_cm, [m, n], [1, lda])
    x_t = create_acl_tensor(x_mat, [cols, 1], [1, cols])
    y_t = create_acl_tensor(y_mat, [rows, 1], [1, rows])
    out_t = create_acl_tensor(out_mat, [rows, 1], [1, rows])

    alpha_s = create_acl_scalar(float(alpha), ACL_DOUBLE)
    beta_s = create_acl_scalar(float(beta), ACL_DOUBLE)

    transA = ctypes.c_int64(do_trans)
    transB = ctypes.c_int64(0)
    cubeMathType = ctypes.c_int8(0)

    two_stage_launch('aclnnGemmGetWorkspaceSize', 'aclnnGemm',
                     [A_t, x_t, y_t, alpha_s, beta_s, transA, transB,
                      out_t, cubeMathType])

    destroy_acl_tensor(A_t)
    destroy_acl_tensor(x_t)
    destroy_acl_tensor(y_t)
    destroy_acl_tensor(out_t)
    destroy_acl_scalar(alpha_s)
    destroy_acl_scalar(beta_s)

    # Write result back to strided y
    y[::incy][:rows] = out_mat.reshape(-1)
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    m, n = 5, 4
    alpha, beta = 1.5, 0.75

    A = torch.randn(m, n, dtype=torch.float64, device=device)
    x = torch.randn(n, dtype=torch.float64, device=device)
    y = torch.randn(m, dtype=torch.float64, device=device)
    y_clone = y.clone()

    result = cublasDgemv_v2(0, m, n, alpha, A, m, x, 1, beta, y, 1)
    expected = alpha * (A @ x) + beta * y_clone
    torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-7, atol=1e-7)
    print("pass: N (no transpose)")

    # Transpose test
    A2 = torch.randn(m, n, dtype=torch.float64, device=device)
    x2 = torch.randn(m, dtype=torch.float64, device=device)
    y2 = torch.randn(n, dtype=torch.float64, device=device)
    y2_clone = y2.clone()

    result2 = cublasDgemv_v2(1, m, n, alpha, A2, m, x2, 1, beta, y2, 1)
    expected2 = alpha * (A2.t() @ x2) + beta * y2_clone
    torch.testing.assert_close(result2.cpu(), expected2.cpu(), rtol=1e-7, atol=1e-7)
    print("pass: T (transpose)")

    print("\ncublasDgemv_v2 all tests passed")
