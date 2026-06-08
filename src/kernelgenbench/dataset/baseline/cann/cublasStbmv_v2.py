import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass


def cublasStbmv_v2(uplo, trans, diag, n, k, A, lda, x, incx):
    """CANN baseline for cublasStbmv_v2 — PyTorch fallback.
    CANN has no band-matrix primitive; uses PyTorch ops.

    x = op(A_band) * x  (float32, triangular band matrix-vector multiply)
    A is in banded storage format.
    """
    # Parse transpose
    if isinstance(trans, str):
        trans_int = 0 if trans.upper() == 'N' else 1
    else:
        trans_int = 1 if trans in (1, 2) else 0

    is_upper = (uplo == 1)
    is_unit = (diag == 1)

    # Reconstruct full triangular band matrix from banded storage
    A_flat = A.reshape(-1)
    AB = torch.as_strided(A_flat, (lda, n), (1, lda)).contiguous()

    A_full = torch.zeros(n, n, dtype=A.dtype, device=A.device)

    if is_upper:
        for j in range(n):
            for i in range(max(0, j - k), j + 1):
                A_full[i, j] = AB[k + i - j, j]
    else:
        for j in range(n):
            for i in range(j, min(n, j + k + 1)):
                A_full[i, j] = AB[i - j, j]

    if is_unit:
        A_full.fill_diagonal_(1.0)

    xs = x[::incx][:n].contiguous()

    if trans_int == 0:
        result = A_full @ xs
    else:
        result = A_full.T @ xs

    x[::incx][:n] = result
    return x


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    n, k = 6, 2
    lda = k + 1
    uplo = 1  # upper
    trans = 0  # no transpose
    diag = 0   # non-unit

    A_full = torch.zeros(n, n, dtype=torch.float32, device=device)
    rand_vals = torch.randn(n, n, dtype=torch.float32, device=device)
    for j in range(n):
        for i in range(max(0, j - k), j + 1):
            A_full[i, j] = rand_vals[i, j]

    AB = torch.zeros(lda, n, dtype=torch.float32, device=device)
    for j in range(n):
        for i in range(max(0, j - k), j + 1):
            AB[k + i - j, j] = A_full[i, j]

    AB_cm = AB.t().contiguous()
    x = torch.randn(n, dtype=torch.float32, device=device)
    x_ref = x.clone()

    result = cublasStbmv_v2(uplo, trans, diag, n, k, AB_cm, lda, x, 1)
    expected = A_full @ x_ref
    torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-5)
    print("cublasStbmv_v2 test passed")
