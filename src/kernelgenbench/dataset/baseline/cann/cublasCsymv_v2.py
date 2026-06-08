import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass


def _complex_mv(A, x):
    """Matrix-vector multiply for complex tensors on devices without complex matmul."""
    if not A.is_complex():
        return A @ x
    Ar, Ai = A.real.contiguous(), A.imag.contiguous()
    xr, xi = x.real.contiguous(), x.imag.contiguous()
    return torch.complex(Ar @ xr - Ai @ xi, Ar @ xi + Ai @ xr)


def cublasCsymv_v2(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    """CANN baseline for cublasCsymv_v2 — PyTorch fallback.
    CANN has no symv primitive; uses PyTorch ops.

    y = alpha * A * x + beta * y  where A is symmetric (complex64).
    """
    A_flat = A.reshape(-1)
    A_cm = torch.as_strided(A_flat, (n, n), (1, lda))

    xs = x[::incx][:n].contiguous()
    ys = y[::incy][:n].contiguous()

    dtype = y.dtype
    dev = y.device
    alpha_t = torch.tensor(alpha, dtype=dtype, device=dev) if not isinstance(alpha, torch.Tensor) else alpha
    beta_t = torch.tensor(beta, dtype=dtype, device=dev) if not isinstance(beta, torch.Tensor) else beta

    result = alpha_t * _complex_mv(A_cm.contiguous(), xs) + beta_t * ys
    y[::incy][:n] = result
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    n = 5
    M = torch.randn(n, n, device=device, dtype=torch.complex64)
    A = 0.5 * (M + M.t())
    x = torch.randn(n, device=device, dtype=torch.complex64)
    y = torch.randn(n, device=device, dtype=torch.complex64)
    A_ref, x_ref, y_ref = A.clone(), x.clone(), y.clone()

    alpha = complex(1.25, -0.5)
    beta = complex(-0.3, 0.75)

    y_out = cublasCsymv_v2(0, n, alpha, A, n, x, 1, beta, y, 1)
    expected = alpha * (A_ref @ x_ref) + beta * y_ref
    torch.testing.assert_close(y_out.cpu(), expected.cpu(), rtol=1e-5, atol=1e-5)
    print("cublasCsymv_v2 test passed")
