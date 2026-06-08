import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass


def cublasDsbmv_v2(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    """CANN baseline for cublasDsbmv_v2 — PyTorch fallback.
    CANN has no band-matrix primitive; uses PyTorch ops.

    y = alpha * A_band * x + beta * y  (float64)
    A is in banded storage format.
    """
    # Reconstruct full symmetric band matrix from banded storage
    A_flat = A.reshape(-1)
    # Column-major banded storage: AB(lda, n)
    AB = torch.as_strided(A_flat, (lda, n), (1, lda)).contiguous()

    A_full = torch.zeros(n, n, dtype=A.dtype, device=A.device)
    is_upper = (uplo == 1)

    if is_upper:
        for j in range(n):
            for i in range(max(0, j - k), j + 1):
                val = AB[k + i - j, j]
                A_full[i, j] = val
                A_full[j, i] = val
    else:
        for j in range(n):
            for i in range(j, min(n, j + k + 1)):
                val = AB[i - j, j]
                A_full[i, j] = val
                A_full[j, i] = val

    xs = x[::incx][:n].contiguous()
    ys = y[::incy][:n].contiguous()

    result = alpha * (A_full @ xs) + beta * ys
    y[::incy][:n] = result
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    n, k = 8, 2
    lda = k + 1
    uplo = 1  # upper

    A_full = torch.zeros(n, n, dtype=torch.float64, device=device)
    for d in range(k + 1):
        vals = torch.randn(n - d, dtype=torch.float64, device=device)
        for i in range(n - d):
            A_full[i, i + d] = vals[i]
            A_full[i + d, i] = vals[i]

    AB = torch.zeros(lda, n, dtype=torch.float64, device=device)
    for j in range(n):
        for i in range(max(0, j - k), j + 1):
            AB[k + i - j, j] = A_full[i, j]

    AB_cm = AB.t().contiguous()
    x = torch.randn(n, dtype=torch.float64, device=device)
    y = torch.randn(n, dtype=torch.float64, device=device)
    y_ref = y.clone()

    alpha, beta = 1.25, -0.75
    y_out = cublasDsbmv_v2(uplo, n, k, alpha, AB_cm, lda, x, 1, beta, y, 1)
    expected = alpha * (A_full @ x) + beta * y_ref
    torch.testing.assert_close(y_out, expected, rtol=1e-5, atol=1e-5)
    print("cublasDsbmv_v2 test passed")
