import ctypes
import torch

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_scalar, destroy_acl_scalar,
                           two_stage_launch, torch_dtype_to_acl,
                           ACL_COMPLEX128)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_scalar, destroy_acl_scalar,
        two_stage_launch, torch_dtype_to_acl,
        ACL_COMPLEX128)


def cublasZgerc_v2(m, n, alpha, x, incx, y, incy, A, lda):
    """CANN baseline for cublasZgerc_v2 — ctypes aclnn C API.
    Conjugated rank-1 update: A = alpha * x * conj(y)^T + A (complex128).

    Steps:
      1. aclnnConj(y) -> conj_y
      2. aclnnGer(x, conj_y) -> ger_out
      3. aclnnMuls(ger_out, alpha) -> scaled
      4. aclnnAdd(A, scaled, 1.0) -> A
    """
    xs = x[::incx][:m].contiguous()
    ys = y[::incy][:n].contiguous()

    # Step 1: conj(y)
    conj_y = torch.empty_like(ys)
    ys_t = create_acl_tensor(ys, [n], [1])
    conj_y_t = create_acl_tensor(conj_y, [n], [1])
    two_stage_launch('aclnnConjGetWorkspaceSize', 'aclnnConj',
                     [ys_t, conj_y_t])
    destroy_acl_tensor(ys_t)
    destroy_acl_tensor(conj_y_t)

    # Step 2: ger(x, conj_y) -> ger_out
    ger_out = torch.empty(m, n, dtype=torch.complex128, device=A.device)
    xs_t = create_acl_tensor(xs, [m], [1])
    conj_y_t2 = create_acl_tensor(conj_y, [n], [1])
    ger_out_t = create_acl_tensor(ger_out, [m, n], [n, 1])
    two_stage_launch('aclnnGerGetWorkspaceSize', 'aclnnGer',
                     [xs_t, conj_y_t2, ger_out_t])
    destroy_acl_tensor(xs_t)
    destroy_acl_tensor(conj_y_t2)
    destroy_acl_tensor(ger_out_t)

    # Step 3: muls(ger_out, alpha) -> scaled
    scaled = torch.empty_like(ger_out)
    alpha_s = create_acl_scalar(alpha, ACL_COMPLEX128)
    ger_out_t2 = create_acl_tensor(ger_out, [m, n], [n, 1])
    scaled_t = create_acl_tensor(scaled, [m, n], [n, 1])
    two_stage_launch('aclnnMulsGetWorkspaceSize', 'aclnnMuls',
                     [ger_out_t2, alpha_s, scaled_t])
    destroy_acl_tensor(ger_out_t2)
    destroy_acl_tensor(scaled_t)
    destroy_acl_scalar(alpha_s)

    # Step 4: add(A, scaled, 1.0) -> A
    # A is column-major with lda; view as (m, n) with stride (1, lda)
    A_flat = A.reshape(-1)
    A_view = torch.as_strided(A_flat, [m, n], [1, lda])
    A_contig = A_view.contiguous()
    one_s = create_acl_scalar(complex(1.0, 0.0), ACL_COMPLEX128)
    A_t = create_acl_tensor(A_contig, [m, n], [n, 1])
    scaled_t2 = create_acl_tensor(scaled, [m, n], [n, 1])
    out_t = create_acl_tensor(A_contig, [m, n], [n, 1])
    two_stage_launch('aclnnAddGetWorkspaceSize', 'aclnnAdd',
                     [A_t, scaled_t2, one_s, out_t])
    destroy_acl_tensor(A_t)
    destroy_acl_tensor(scaled_t2)
    destroy_acl_tensor(out_t)
    destroy_acl_scalar(one_s)

    # Write back to column-major A
    A_view.copy_(A_contig)
    return A


if __name__ == "__main__":
    torch.manual_seed(42)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    m, n = 4, 3
    alpha = complex(0.5, -0.3)
    x = torch.randn(m, dtype=torch.complex128, device=device)
    y = torch.randn(n, dtype=torch.complex128, device=device)
    A = torch.randn(m, n, dtype=torch.complex128, device=device)
    A_clone = A.clone()

    result = cublasZgerc_v2(m, n, alpha, x, 1, y, 1, A, m)
    expected = A_clone + alpha * x.unsqueeze(1) * y.conj().unsqueeze(0)
    torch.testing.assert_close(result.cpu(), expected.cpu(), rtol=1e-7, atol=1e-7)
    print("pass: basic Zgerc")

    # strided test
    x2 = torch.randn(20, dtype=torch.complex128, device=device)
    y2 = torch.randn(15, dtype=torch.complex128, device=device)
    A2 = torch.randn(3, 4, dtype=torch.complex128, device=device)
    A2_clone = A2.clone()
    result2 = cublasZgerc_v2(3, 4, complex(1.0, 0.0), x2, 3, y2, 2, A2, 3)
    expected2 = A2_clone + x2[::3][:3].unsqueeze(1) * y2[::2][:4].conj().unsqueeze(0)
    torch.testing.assert_close(result2.cpu(), expected2.cpu(), rtol=1e-7, atol=1e-7)
    print("pass: strided Zgerc")

    print("\ncublasZgerc_v2 all tests passed")
