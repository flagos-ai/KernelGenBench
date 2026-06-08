import ctypes
import torch

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_scalar, destroy_acl_scalar,
                           two_stage_launch, map_transpose,
                           ACL_COMPLEX128)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_scalar, destroy_acl_scalar,
        two_stage_launch, map_transpose,
        ACL_COMPLEX128)


def cublasZgemvBatched(trans, m, n, alpha, Aarray, lda, xarray, incx,
                       beta, yarray, incy, batchCount):
    """CANN baseline for cublasZgemvBatched — ctypes aclnn C API.
    y[i] = alpha * op(A[i]) * x[i] + beta * y[i]  (complex128, pointer-array batched)

    Uses aclnnBatchMatMul by stacking into 3D tensors and reshaping vectors.
    Then aclnnMuls for alpha scaling and aclnnAdd for beta*y accumulation.
    """
    do_trans = map_transpose(trans)

    if do_trans:
        rows, cols = n, m
    else:
        rows, cols = m, n

    # Build column-major views of A
    A_views = []
    for i in range(batchCount):
        a_flat = Aarray[i].reshape(-1)
        a_cm = torch.as_strided(a_flat, [m, n], [1, lda])
        if do_trans:
            A_views.append(a_cm.t().contiguous())
        else:
            A_views.append(a_cm.contiguous())
    A_3d = torch.stack(A_views)

    x_batch = torch.stack([xarray[i][::incx][:cols] for i in range(batchCount)])
    y_batch = torch.stack([yarray[i][::incy][:rows] for i in range(batchCount)])

    x_3d = x_batch.unsqueeze(-1).contiguous()
    y_3d = y_batch.unsqueeze(-1).contiguous()
    out_3d = torch.empty(batchCount, rows, 1, dtype=torch.complex128, device=A_3d.device)

    # aclnnBatchMatMul
    A_t = create_acl_tensor(A_3d, [batchCount, rows, cols],
                            [rows * cols, cols, 1])
    x_t = create_acl_tensor(x_3d, [batchCount, cols, 1],
                            [cols, 1, 1])
    out_t = create_acl_tensor(out_3d, [batchCount, rows, 1],
                              [rows, 1, 1])
    cubeMathType = ctypes.c_int8(0)

    two_stage_launch('aclnnBatchMatMulGetWorkspaceSize', 'aclnnBatchMatMul',
                     [A_t, x_t, out_t, cubeMathType])
    destroy_acl_tensor(A_t)
    destroy_acl_tensor(x_t)
    destroy_acl_tensor(out_t)

    # Scale by alpha
    scaled = torch.empty_like(out_3d)
    alpha_s = create_acl_scalar(complex(alpha), ACL_COMPLEX128)
    out_t2 = create_acl_tensor(out_3d, [batchCount, rows, 1],
                               [rows, 1, 1])
    scaled_t = create_acl_tensor(scaled, [batchCount, rows, 1],
                                 [rows, 1, 1])
    two_stage_launch('aclnnMulsGetWorkspaceSize', 'aclnnMuls',
                     [out_t2, alpha_s, scaled_t])
    destroy_acl_tensor(out_t2)
    destroy_acl_tensor(scaled_t)
    destroy_acl_scalar(alpha_s)

    # beta * y
    beta_y = torch.empty_like(y_3d)
    beta_s = create_acl_scalar(complex(beta), ACL_COMPLEX128)
    y_t = create_acl_tensor(y_3d, [batchCount, rows, 1],
                            [rows, 1, 1])
    beta_y_t = create_acl_tensor(beta_y, [batchCount, rows, 1],
                                 [rows, 1, 1])
    two_stage_launch('aclnnMulsGetWorkspaceSize', 'aclnnMuls',
                     [y_t, beta_s, beta_y_t])
    destroy_acl_tensor(y_t)
    destroy_acl_tensor(beta_y_t)
    destroy_acl_scalar(beta_s)

    # Add: beta_y + scaled -> result
    result = torch.empty_like(beta_y)
    one_s = create_acl_scalar(complex(1.0, 0.0), ACL_COMPLEX128)
    beta_y_t2 = create_acl_tensor(beta_y, [batchCount, rows, 1],
                                  [rows, 1, 1])
    scaled_t2 = create_acl_tensor(scaled, [batchCount, rows, 1],
                                  [rows, 1, 1])
    result_t = create_acl_tensor(result, [batchCount, rows, 1],
                                 [rows, 1, 1])
    two_stage_launch('aclnnAddGetWorkspaceSize', 'aclnnAdd',
                     [beta_y_t2, scaled_t2, one_s, result_t])
    destroy_acl_tensor(beta_y_t2)
    destroy_acl_tensor(scaled_t2)
    destroy_acl_tensor(result_t)
    destroy_acl_scalar(one_s)

    # Write results back
    res_2d = result.squeeze(-1)
    for i in range(batchCount):
        yarray[i][::incy][:rows] = res_2d[i]

    return yarray


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    batchCount = 3
    m, n = 4, 5
    alpha = complex(1.5, 0.5)
    beta = complex(0.75, -0.25)

    Aarray = [torch.randn(m, n, dtype=torch.complex128, device=device) for _ in range(batchCount)]
    xarray = [torch.randn(n, dtype=torch.complex128, device=device) for _ in range(batchCount)]
    yarray = [torch.randn(m, dtype=torch.complex128, device=device) for _ in range(batchCount)]
    y_clones = [yi.clone() for yi in yarray]

    result = cublasZgemvBatched(0, m, n, alpha, Aarray, m, xarray, 1,
                                beta, yarray, 1, batchCount)
    for i in range(batchCount):
        expected = alpha * (Aarray[i] @ xarray[i]) + beta * y_clones[i]
        torch.testing.assert_close(result[i].cpu(), expected.cpu(), rtol=1e-7, atol=1e-7)
    print("pass: batched gemv N (complex128)")

    print("\ncublasZgemvBatched all tests passed")
