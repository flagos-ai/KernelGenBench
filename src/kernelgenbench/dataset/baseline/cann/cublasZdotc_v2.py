import torch

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           two_stage_launch)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor, two_stage_launch)


def cublasZdotc_v2(n, x, incx, y, incy, result):
    """CANN baseline for cublasZdotc_v2 — ctypes aclnnDot C API.
    Computes result = conj(x)^T . y (conjugate dot product, complex128).
    Uses aclnnConj on x first, then aclnnDot.
    """
    xs = x[::incx][:n].contiguous()
    ys = y[::incy][:n].contiguous()
    res = result.contiguous()

    # Step 1: conj(x)
    xs_conj = torch.empty_like(xs)
    shape_n = [n]
    stride_1 = [1]
    shape_1 = [1]

    xs_t = create_acl_tensor(xs, shape_n, stride_1)
    xs_conj_t = create_acl_tensor(xs_conj, shape_n, stride_1)

    two_stage_launch('aclnnConjGetWorkspaceSize', 'aclnnConj',
                     [xs_t, xs_conj_t])

    destroy_acl_tensor(xs_t)
    destroy_acl_tensor(xs_conj_t)

    # Step 2: dot(conj(x), y)
    xc_t = create_acl_tensor(xs_conj, shape_n, stride_1)
    y_t = create_acl_tensor(ys, shape_n, stride_1)
    out_t = create_acl_tensor(res, shape_1, stride_1)

    two_stage_launch('aclnnDotGetWorkspaceSize', 'aclnnDot',
                     [xc_t, y_t, out_t])

    destroy_acl_tensor(xc_t)
    destroy_acl_tensor(y_t)
    destroy_acl_tensor(out_t)

    result.copy_(res)
    return result


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    n = 5
    incx, incy = 2, 3
    x = torch.randn(30, dtype=torch.complex128, device=device)
    y = torch.randn(30, dtype=torch.complex128, device=device)
    result = torch.empty(1, dtype=torch.complex128, device=device)

    out = cublasZdotc_v2(n, x, incx, y, incy, result)
    x_slice = x[::incx][:n]
    y_slice = y[::incy][:n]
    expected = (x_slice.conj() * y_slice).sum()
    torch.testing.assert_close(out[0], expected, rtol=1e-5, atol=1e-5)
    print("✓ cublasZdotc_v2 test passed")
