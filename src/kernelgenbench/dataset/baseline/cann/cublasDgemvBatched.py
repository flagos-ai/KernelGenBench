import ctypes
import torch

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_scalar, destroy_acl_scalar,
                           two_stage_launch, map_transpose,
                           ACL_DOUBLE)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_scalar, destroy_acl_scalar,
        two_stage_launch, map_transpose,
        ACL_DOUBLE)


def cublasDgemvBatched(trans, m, n, alpha, Aarray, lda, xarray, incx,
                       beta, yarray, incy, batchCount):
    """CANN baseline for cublasDgemvBatched — ctypes aclnn C API.
    y[i] = alpha * op(A[i]) * x[i] + beta * y[i]  (float64, pointer-array batched)

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
    A_3d = torch.stack(A_views)  # (batch, rows, cols)

    x_batch = torch.stack([xarray[i][::incx][:cols] for i in range(batchCount)])
    y_batch = torch.stack([yarray[i][::incy][:rows] for i in range(batchCount)])

    x_3d = x_batch.unsqueeze(-1).contiguous()  # (batch, cols, 1)
    y_3d = y_batch.unsqueeze(-1).contiguous()  # (batch, rows, 1)
    out_3d = torch.empty(batchCount, rows, 1, dtype=torch.float64, device=A_3d.device)

    # aclnnBatchMatMul(self, mat2, out, cubeMathType)
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
    alpha_s = create_acl_scalar(float(alpha), ACL_DOUBLE)
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
    beta_s = create_acl_scalar(float(beta), ACL_DOUBLE)
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
    one_s = create_acl_scalar(1.0, ACL_DOUBLE)
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
    alpha, beta = 1.5, 0.75

    Aarray = [torch.randn(m, n, dtype=torch.float64, device=device) for _ in range(batchCount)]
    xarray = [torch.randn(n, dtype=torch.float64, device=device) for _ in range(batchCount)]
    yarray = [torch.randn(m, dtype=torch.float64, device=device) for _ in range(batchCount)]
    y_clones = [yi.clone() for yi in yarray]

    result = cublasDgemvBatched(0, m, n, alpha, Aarray, m, xarray, 1,
                                beta, yarray, 1, batchCount)
    for i in range(batchCount):
        expected = alpha * (Aarray[i] @ xarray[i]) + beta * y_clones[i]
        torch.testing.assert_close(result[i], expected, rtol=1e-7, atol=1e-7)
    print("pass: batched gemv N")

    print("\ncublasDgemvBatched all tests passed")
