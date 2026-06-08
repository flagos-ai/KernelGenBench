import torch

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           two_stage_launch, torch_dtype_to_acl)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        two_stage_launch, torch_dtype_to_acl)


def cublasDcopy_v2(n, x, incx, y, incy):
    """CANN baseline for cublasDcopy_v2 — ctypes aclnnCopy C API.
    Copies n elements from x (stride incx) into y (stride incy).
    """
    # Extract strided views as contiguous tensors
    x_src = x[::incx][:n].contiguous()
    y_dst = y[::incy][:n].contiguous()

    shape = [n]
    stride = [1]

    src_t = create_acl_tensor(x_src, shape, stride)
    dst_t = create_acl_tensor(y_dst, shape, stride)
    out_t = create_acl_tensor(y_dst, shape, stride)

    # aclnnCopyGetWorkspaceSize(dst, src, out, &ws_size, &executor)
    two_stage_launch('aclnnCopyGetWorkspaceSize', 'aclnnCopy',
                     [dst_t, src_t, out_t])

    destroy_acl_tensor(src_t)
    destroy_acl_tensor(dst_t)
    destroy_acl_tensor(out_t)

    # Write back to strided y
    y[::incy][:n] = y_dst
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    n = 100
    x = torch.randn(n, dtype=torch.float64, device=device)
    y = torch.randn(n, dtype=torch.float64, device=device)

    result = cublasDcopy_v2(n, x, 1, y, 1)
    torch.testing.assert_close(result, x, rtol=1e-7, atol=1e-7)
    print("pass: basic copy")

    x2 = torch.randn(200, dtype=torch.float64, device=device)
    y2 = torch.randn(300, dtype=torch.float64, device=device)

    result2 = cublasDcopy_v2(50, x2, 2, y2, 3)
    torch.testing.assert_close(result2[::3][:50], x2[::2][:50], rtol=1e-7, atol=1e-7)
    print("pass: strided copy")

    print("\n✓ cublasDcopy_v2 all tests passed")
