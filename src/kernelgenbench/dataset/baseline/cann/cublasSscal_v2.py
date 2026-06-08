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


def cublasSscal_v2(n, alpha, x, incx):
    """CANN baseline for cublasSscal_v2 — ctypes aclnnMuls C API.
    Computes x = alpha * x for n float32 elements with stride.
    """
    # Extract strided slice as contiguous tensor
    x_slice = x[::incx][:n].contiguous()
    out_buf = x_slice.clone()

    shape = [n]
    stride = [1]

    x_t = create_acl_tensor(x_slice, shape, stride)
    out_t = create_acl_tensor(out_buf, shape, stride)
    alpha_s = create_acl_scalar(float(alpha), ACL_FLOAT)

    # aclnnMulsGetWorkspaceSize(self, other, out, &ws_size, &executor)
    two_stage_launch('aclnnMulsGetWorkspaceSize', 'aclnnMuls',
                     [x_t, alpha_s, out_t])

    destroy_acl_tensor(x_t)
    destroy_acl_tensor(out_t)
    destroy_acl_scalar(alpha_s)

    # Write back to strided x
    x[::incx][:n] = out_buf
    return x


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    x = torch.randn(10, dtype=torch.float32, device=device)
    x_clone = x.clone()
    result = cublasSscal_v2(10, 2.5, x, 1)
    expected = x_clone * 2.5
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("pass: unit stride")

    # strided
    x2 = torch.randn(100, dtype=torch.float32, device=device)
    x2_clone = x2.clone()
    result2 = cublasSscal_v2(20, -0.5, x2, 3)
    expected2 = x2_clone.clone()
    expected2[::3][:20] *= -0.5
    torch.testing.assert_close(result2, expected2, rtol=1e-5, atol=1e-5)
    print("pass: strided")

    print("\ncublasSscal_v2 all tests passed")
