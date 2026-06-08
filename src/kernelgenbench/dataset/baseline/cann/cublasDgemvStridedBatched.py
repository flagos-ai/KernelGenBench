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


def cublasDgemvStridedBatched(trans, m, n, alpha, A, lda, strideA,
                              x, incx, stridex, beta, y, incy, stridey,
                              batchCount):
    """CANN baseline for cublasDgemvStridedBatched — ctypes aclnn C API.
    y[i] = alpha * op(A[i]) * x[i] + beta * y[i]  (float64, strided batched)

    Uses aclnnBatchMatMul by constructing 3D tensors from strided data.
    Then aclnnMuls for alpha scaling and aclnnAdd for beta*y accumulation.
    """
    do_trans = map_transpose(trans)

    if do_trans:
        rows, cols = n, m
    else:
        rows, cols = m, n

    A_flat = A.reshape(-1)
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)

    A_views = []
    x_list = []
    y_list = []
    for i in range(batchCount):
        a_off = i * strideA
        a_cm = torch.as_strided(A_flat[a_off:], [m, n], [1, lda])
        if do_trans:
            A_views.append(a_cm.t().contiguous())
        else:
            A_views.append(a_cm.contiguous())
        x_off = i * stridex
        x_list.append(x_flat[x_off:][::incx][:cols].contiguous())
        y_off = i * stridey
        y_list.append(y_flat[y_off:][::incy][:rows].contiguous())

    A_3d = torch.stack(A_views)
    x_3d = torch.stack(x_list).unsqueeze(-1).contiguous()
    y_3d = torch.stack(y_list).unsqueeze(-1).contiguous()
    out_3d = torch.empty(batchCount, rows, 1, dtype=torch.float64, device=A_3d.device)

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

    # Write results back to strided y
    res_2d = result.squeeze(-1)
    for i in range(batchCount):
        y_off = i * stridey
        y_flat[y_off:][::incy][:rows] = res_2d[i]

    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    batchCount = 3
    m, n = 4, 5
    alpha, beta = 1.5, 0.75

    A = torch.randn(batchCount, m, n, dtype=torch.float64, device=device).contiguous()
    x = torch.randn(batchCount, n, dtype=torch.float64, device=device).contiguous()
    y = torch.randn(batchCount, m, dtype=torch.float64, device=device).contiguous()
    y_clone = y.clone()

    strideA = m * n
    stridex = n
    stridey = m

    result = cublasDgemvStridedBatched(0, m, n, alpha, A, m, strideA,
                                       x, 1, stridex, beta, y, 1, stridey,
                                       batchCount)
    for i in range(batchCount):
        expected = alpha * (A[i] @ x[i]) + beta * y_clone[i]
        torch.testing.assert_close(result.reshape(batchCount, m)[i],
                                   expected, rtol=1e-7, atol=1e-7)
    print("pass: strided batched gemv N")

    print("\ncublasDgemvStridedBatched all tests passed")
