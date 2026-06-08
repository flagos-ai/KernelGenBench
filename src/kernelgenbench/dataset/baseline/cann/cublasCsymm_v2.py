import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    pass


def _complex_mm(A, B):
    """Matrix multiply for complex tensors on devices without complex matmul."""
    if not A.is_complex():
        return A @ B
    Ar, Ai = A.real.contiguous(), A.imag.contiguous()
    Br, Bi = B.real.contiguous(), B.imag.contiguous()
    return torch.complex(Ar @ Br - Ai @ Bi, Ar @ Bi + Ai @ Br)


def cublasCsymm_v2(side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc):
    """CANN baseline for cublasCsymm_v2 — PyTorch fallback.
    CANN has no symm primitive; uses PyTorch ops.

    C = alpha * A * B + beta * C  (side=LEFT) or
    C = alpha * B * A + beta * C  (side=RIGHT)
    where A is symmetric (complex64).
    Data is in column-major layout.
    """
    A_flat = A.reshape(-1)
    B_flat = B.reshape(-1)
    C_flat = C.reshape(-1)

    is_left = (side == 0)  # CUBLAS_SIDE_LEFT
    dim_a = m if is_left else n

    A_cm = torch.as_strided(A_flat, (dim_a, dim_a), (1, lda))
    B_cm = torch.as_strided(B_flat, (m, n), (1, ldb))
    C_cm = torch.as_strided(C_flat, (m, n), (1, ldc))

    dtype = C.dtype
    dev = C.device
    alpha_t = torch.tensor(alpha, dtype=dtype, device=dev) if not isinstance(alpha, torch.Tensor) else alpha
    beta_t = torch.tensor(beta, dtype=dtype, device=dev) if not isinstance(beta, torch.Tensor) else beta

    A_cont = A_cm.contiguous()
    B_cont = B_cm.contiguous()

    if is_left:
        result = alpha_t * _complex_mm(A_cont, B_cont) + beta_t * C_cm
    else:
        result = alpha_t * _complex_mm(B_cont, A_cont) + beta_t * C_cm

    C_cm.copy_(result)
    return C


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    m, n = 3, 4
    A_rand = torch.randn(m, m, dtype=torch.complex64, device=device)
    A_sym = A_rand + A_rand.t()
    B_rm = torch.randn(m, n, dtype=torch.complex64, device=device)
    C_rm = torch.randn(m, n, dtype=torch.complex64, device=device)

    alpha = 0.7 + 0.3j
    beta = -0.2 + 0.5j
    A_ref, B_ref, C_ref = A_sym.clone(), B_rm.clone(), C_rm.clone()

    A_cm = A_sym.t().contiguous()
    B_cm = B_rm.t().contiguous()
    C_cm = C_rm.t().contiguous()

    out = cublasCsymm_v2(0, 1, m, n, alpha, A_cm, m, B_cm, m, beta, C_cm, m)
    result_rm = out.t().contiguous()
    expected = alpha * (A_ref @ B_ref) + beta * C_ref
    torch.testing.assert_close(result_rm.cpu(), expected.cpu(), rtol=1e-4, atol=1e-4)
    print("cublasCsymm_v2 test passed")
