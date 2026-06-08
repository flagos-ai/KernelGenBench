import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass


def cublasDsyr2_v2(uplo, n, alpha, x, incx, y, incy, A, lda):
    """CANN baseline for cublasDsyr2_v2 — PyTorch fallback.
    CANN has no syr2 primitive; uses PyTorch ops.

    A = alpha * (x * y^T + y * x^T) + A  (float64, symmetric rank-2 update)
    Only the upper or lower triangle is updated.
    Data is in column-major layout.
    """
    A_flat = A.reshape(-1)
    A_cm = torch.as_strided(A_flat, (n, n), (1, lda))

    xs = x[::incx][:n].contiguous()
    ys = y[::incy][:n].contiguous()

    update = alpha * (xs.unsqueeze(1) @ ys.unsqueeze(0) + ys.unsqueeze(1) @ xs.unsqueeze(0))

    is_upper = (uplo == 1)
    if is_upper:
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=A.device))
    else:
        mask = torch.tril(torch.ones(n, n, dtype=torch.bool, device=A.device))

    A_cm_cont = A_cm.contiguous()
    A_cm_cont[mask] = (A_cm_cont + update)[mask]
    A_cm.copy_(A_cm_cont)
    return A


if __name__ == "__main__":
    torch.manual_seed(1234)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    n = 5
    alpha = 0.75

    A = torch.randn(n, n, dtype=torch.float64, device=device)
    x = torch.randn(n, dtype=torch.float64, device=device)
    y = torch.randn(n, dtype=torch.float64, device=device)
    A0, x0, y0 = A.clone(), x.clone(), y.clone()

    result = cublasDsyr2_v2(1, n, alpha, x, 1, y, 1, A, n)

    A_col0 = A0.t().contiguous()
    M_col = alpha * (x0.view(n, 1) @ y0.view(1, n) + y0.view(n, 1) @ x0.view(1, n))
    A_col_expected = torch.triu(A_col0 + M_col) + torch.tril(A_col0, -1)
    expected = A_col_expected.t().contiguous()

    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("cublasDsyr2_v2 test passed")
