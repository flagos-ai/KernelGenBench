import torch

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           two_stage_launch, torch_dtype_to_acl)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        two_stage_launch, torch_dtype_to_acl)


def cublasSdot_v2(n, x, incx, y, incy, result):
    """CANN baseline for cublasSdot_v2 — ctypes aclnnDot C API.
    Computes dot product of n float32 elements from x (stride incx)
    and y (stride incy). Result stored in 1-element tensor `result`.
    """
    # Extract strided slices as contiguous tensors
    x_slice = x[::incx][:n].contiguous()
    y_slice = y[::incy][:n].contiguous()
    out_buf = result.contiguous()

    shape_1d = [n]
    stride_1d = [1]
    shape_out = list(out_buf.shape) if out_buf.dim() > 0 else [1]
    stride_out = list(out_buf.stride()) if out_buf.dim() > 0 else [1]

    x_t = create_acl_tensor(x_slice, shape_1d, stride_1d)
    y_t = create_acl_tensor(y_slice, shape_1d, stride_1d)
    out_t = create_acl_tensor(out_buf, shape_out, stride_out)

    # aclnnDotGetWorkspaceSize(self, tensor, out, &ws_size, &executor)
    two_stage_launch('aclnnDotGetWorkspaceSize', 'aclnnDot',
                     [x_t, y_t, out_t])

    destroy_acl_tensor(x_t)
    destroy_acl_tensor(y_t)
    destroy_acl_tensor(out_t)

    # Write back
    result.copy_(out_buf)
    return result


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    n = 100
    x = torch.randn(n, dtype=torch.float32, device=device)
    y = torch.randn(n, dtype=torch.float32, device=device)
    result = torch.zeros(1, dtype=torch.float32, device=device)

    cublasSdot_v2(n, x, 1, y, 1, result)
    expected = torch.dot(x, y)
    torch.testing.assert_close(result.squeeze(), expected, rtol=1e-4, atol=1e-4)
    print("pass: basic dot")

    x2 = torch.randn(200, dtype=torch.float32, device=device)
    y2 = torch.randn(300, dtype=torch.float32, device=device)
    result2 = torch.zeros(1, dtype=torch.float32, device=device)

    cublasSdot_v2(50, x2, 2, y2, 3, result2)
    expected2 = torch.dot(x2[::2][:50], y2[::3][:50])
    torch.testing.assert_close(result2.squeeze(), expected2, rtol=1e-4, atol=1e-4)
    print("pass: strided dot")

    print("\ncublasSdot_v2 all tests passed")
