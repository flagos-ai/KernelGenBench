import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass


def cublasZswap_v2(n, x, incx, y, incy):
    """CANN baseline for cublasZswap_v2 — PyTorch fallback.
    CANN has no swap primitive; uses PyTorch ops.

    Swaps n elements between x (stride incx) and y (stride incy).
    complex128.
    """
    xs = x[::incx][:n]
    ys = y[::incy][:n]

    tmp = xs.clone()
    xs.copy_(ys)
    ys.copy_(tmp)
    return x


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    n = 8
    x = torch.randn(n, dtype=torch.complex128, device=device)
    y = torch.randn(n, dtype=torch.complex128, device=device)
    x0, y0 = x.clone(), y.clone()

    result = cublasZswap_v2(n, x, 1, y, 1)
    torch.testing.assert_close(x, y0, rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(y, x0, rtol=1e-7, atol=1e-7)
    print("cublasZswap_v2 test passed")

    # Strided test
    x2 = torch.randn(20, dtype=torch.complex128, device=device)
    y2 = torch.randn(30, dtype=torch.complex128, device=device)
    x2_ref = x2.clone()
    y2_ref = y2.clone()

    cublasZswap_v2(5, x2, 2, y2, 3)
    torch.testing.assert_close(x2[::2][:5], y2_ref[::3][:5], rtol=1e-7, atol=1e-7)
    torch.testing.assert_close(y2[::3][:5], x2_ref[::2][:5], rtol=1e-7, atol=1e-7)
    print("cublasZswap_v2 strided test passed")
