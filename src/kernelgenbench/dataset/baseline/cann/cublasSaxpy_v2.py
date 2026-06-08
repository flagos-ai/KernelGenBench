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


def cublasSaxpy_v2(n, alpha, x, incx, y, incy):
    """CANN baseline for cublasSaxpy_v2 — ctypes aclnnAdd C API.
    Computes y = alpha * x + y for n float32 elements with strides.
    """
    # Extract strided slices as contiguous tensors
    x_slice = x[::incx][:n].contiguous()
    y_slice = y[::incy][:n].contiguous()
    out_buf = y_slice.clone()

    shape = [n]
    stride = [1]

    y_t = create_acl_tensor(y_slice, shape, stride)
    x_t = create_acl_tensor(x_slice, shape, stride)
    out_t = create_acl_tensor(out_buf, shape, stride)
    alpha_s = create_acl_scalar(float(alpha), ACL_FLOAT)

    # aclnnAddGetWorkspaceSize(self, other, alpha, out, &ws_size, &executor)
    # self=y_slice, other=x_slice, alpha=alpha_scalar, out=out_buf
    two_stage_launch('aclnnAddGetWorkspaceSize', 'aclnnAdd',
                     [y_t, x_t, alpha_s, out_t])

    destroy_acl_tensor(y_t)
    destroy_acl_tensor(x_t)
    destroy_acl_tensor(out_t)
    destroy_acl_scalar(alpha_s)

    # Write back to strided y
    y[::incy][:n] = out_buf
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    n = 100
    alpha = 2.5
    x = torch.randn(n, dtype=torch.float32, device=device)
    y = torch.randn(n, dtype=torch.float32, device=device)
    y_ref = y.clone()

    result = cublasSaxpy_v2(n, alpha, x, 1, y, 1)
    expected = alpha * x + y_ref
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("pass: basic axpy")

    x2 = torch.randn(200, dtype=torch.float32, device=device)
    y2 = torch.randn(300, dtype=torch.float32, device=device)
    y2_ref = y2.clone()

    result2 = cublasSaxpy_v2(50, alpha, x2, 2, y2, 3)
    expected2 = y2_ref.clone()
    expected2[::3][:50] = alpha * x2[::2][:50] + y2_ref[::3][:50]
    torch.testing.assert_close(result2, expected2, rtol=1e-5, atol=1e-5)
    print("pass: strided axpy")

    print("\ncublasSaxpy_v2 all tests passed")
