import torch

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_scalar, destroy_acl_scalar,
                           two_stage_launch, torch_dtype_to_acl,
                           ACL_FLOAT)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_scalar, destroy_acl_scalar,
        two_stage_launch, torch_dtype_to_acl,
        ACL_FLOAT)


def cublasSger_v2(m, n, alpha, x, incx, y, incy, A, lda):
    """CANN baseline for cublasSger_v2 — ctypes aclnnGer + aclnnMuls + aclnnAdd.
    Computes A = alpha * x * y^T + A (rank-1 update), float32.

    Step 1: ger_out = ger(x_slice, y_slice)       -> [m, n]
    Step 2: scaled  = muls(ger_out, alpha)         -> [m, n]
    Step 3: out     = add(A, scaled, alpha=1.0)    -> [m, n]
    """
    # Extract strided slices as contiguous tensors
    x_slice = x[::incx][:m].contiguous()
    y_slice = y[::incy][:n].contiguous()
    A_buf = A.contiguous()

    shape_x = [m]
    stride_x = [1]
    shape_y = [n]
    stride_y = [1]
    shape_2d = [m, n]
    stride_2d = [n, 1]

    # --- Step 1: ger_out = ger(x_slice, y_slice) ---
    ger_out = torch.empty(m, n, dtype=torch.float32, device=x.device)

    x_t = create_acl_tensor(x_slice, shape_x, stride_x)
    y_t = create_acl_tensor(y_slice, shape_y, stride_y)
    ger_out_t = create_acl_tensor(ger_out, shape_2d, stride_2d)

    # aclnnGerGetWorkspaceSize(self, vec2, out, &ws, &exec)
    two_stage_launch('aclnnGerGetWorkspaceSize', 'aclnnGer',
                     [x_t, y_t, ger_out_t])

    destroy_acl_tensor(x_t)
    destroy_acl_tensor(y_t)
    destroy_acl_tensor(ger_out_t)

    # --- Step 2: scaled = muls(ger_out, alpha) ---
    scaled = torch.empty(m, n, dtype=torch.float32, device=x.device)

    ger_t = create_acl_tensor(ger_out, shape_2d, stride_2d)
    scaled_t = create_acl_tensor(scaled, shape_2d, stride_2d)
    alpha_s = create_acl_scalar(float(alpha), ACL_FLOAT)

    # aclnnMulsGetWorkspaceSize(self, other, out, &ws, &exec)
    two_stage_launch('aclnnMulsGetWorkspaceSize', 'aclnnMuls',
                     [ger_t, alpha_s, scaled_t])

    destroy_acl_tensor(ger_t)
    destroy_acl_tensor(scaled_t)
    destroy_acl_scalar(alpha_s)

    # --- Step 3: out = add(A, scaled, alpha=1.0) ---
    out_buf = torch.empty(m, n, dtype=torch.float32, device=x.device)

    A_t = create_acl_tensor(A_buf, shape_2d, stride_2d)
    sc_t = create_acl_tensor(scaled, shape_2d, stride_2d)
    out_t = create_acl_tensor(out_buf, shape_2d, stride_2d)
    one_s = create_acl_scalar(1.0, ACL_FLOAT)

    # aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws, &exec)
    two_stage_launch('aclnnAddGetWorkspaceSize', 'aclnnAdd',
                     [A_t, sc_t, one_s, out_t])

    destroy_acl_tensor(A_t)
    destroy_acl_tensor(sc_t)
    destroy_acl_tensor(out_t)
    destroy_acl_scalar(one_s)

    # Write back
    A.copy_(out_buf)
    return A


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    m, n = 5, 7
    alpha = 0.75
    x = torch.randn(m, dtype=torch.float32, device=device)
    y = torch.randn(n, dtype=torch.float32, device=device)
    A = torch.randn(m, n, dtype=torch.float32, device=device)
    A_clone = A.clone()

    result = cublasSger_v2(m, n, alpha, x, 1, y, 1, A, m)
    expected = A_clone + alpha * x.unsqueeze(1) * y.unsqueeze(0)
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("pass: basic ger")

    # strided
    x2 = torch.randn(20, dtype=torch.float32, device=device)
    y2 = torch.randn(30, dtype=torch.float32, device=device)
    A2 = torch.randn(4, 5, dtype=torch.float32, device=device)
    A2_clone = A2.clone()
    result2 = cublasSger_v2(4, 5, -1.0, x2, 3, y2, 4, A2, 4)
    expected2 = A2_clone + (-1.0) * x2[::3][:4].unsqueeze(1) * y2[::4][:5].unsqueeze(0)
    torch.testing.assert_close(result2, expected2, rtol=1e-5, atol=1e-5)
    print("pass: strided ger")

    print("\ncublasSger_v2 all tests passed")
