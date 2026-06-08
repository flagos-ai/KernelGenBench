import ctypes
import torch

try:
    from ._backend import (create_acl_tensor, destroy_acl_tensor,
                           create_acl_int_array, destroy_acl_int_array,
                           two_stage_launch, torch_dtype_to_acl,
                           ACL_DOUBLE)
except ImportError:
    from kernelgenbench.dataset.baseline.cann._backend import (
        create_acl_tensor, destroy_acl_tensor,
        create_acl_int_array, destroy_acl_int_array,
        two_stage_launch, torch_dtype_to_acl,
        ACL_DOUBLE)


def cublasDasum_v2(n, x, incx, result):
    """CANN baseline for cublasDasum_v2 — ctypes aclnnAbs + aclnnSum C API.
    Computes result = sum(|x_i|) for n float64 elements with stride.

    Step 1: abs_out = abs(x_slice)
    Step 2: sum_out = sum(abs_out, dim=[0], keepDim=False, dtype=ACL_DOUBLE)
    """
    # Extract strided slice as contiguous tensor
    x_slice = x[::incx][:n].contiguous()

    shape_1d = [n]
    stride_1d = [1]

    # --- Step 1: abs_out = abs(x_slice) ---
    abs_out = torch.empty(n, dtype=torch.float64, device=x.device)

    x_t = create_acl_tensor(x_slice, shape_1d, stride_1d)
    abs_t = create_acl_tensor(abs_out, shape_1d, stride_1d)

    # aclnnAbsGetWorkspaceSize(self, out, &ws, &exec)
    two_stage_launch('aclnnAbsGetWorkspaceSize', 'aclnnAbs',
                     [x_t, abs_t])

    destroy_acl_tensor(x_t)
    destroy_acl_tensor(abs_t)

    # --- Step 2: sum_out = sum(abs_out, dim=[0], keepDim=False, dtype=ACL_DOUBLE) ---
    out_buf = result.contiguous()
    shape_out = list(out_buf.shape) if out_buf.dim() > 0 else [1]
    stride_out = list(out_buf.stride()) if out_buf.dim() > 0 else [1]

    abs_t2 = create_acl_tensor(abs_out, shape_1d, stride_1d)
    out_t = create_acl_tensor(out_buf, shape_out, stride_out)
    dim_arr = create_acl_int_array([0])

    # aclnnSumGetWorkspaceSize(self, dimList, keepDim, dtype, out, &ws, &exec)
    two_stage_launch('aclnnSumGetWorkspaceSize', 'aclnnSum',
                     [abs_t2, dim_arr, ctypes.c_bool(False),
                      ctypes.c_int64(ACL_DOUBLE), out_t])

    destroy_acl_tensor(abs_t2)
    destroy_acl_tensor(out_t)
    destroy_acl_int_array(dim_arr)

    # Write back
    result.copy_(out_buf)
    return result


if __name__ == "__main__":
    torch.manual_seed(0)
    device = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cpu'

    n = 100
    x = torch.randn(n, dtype=torch.float64, device=device)
    result = torch.zeros(1, dtype=torch.float64, device=device)

    cublasDasum_v2(n, x, 1, result)
    expected = x.abs().sum()
    torch.testing.assert_close(result.squeeze(), expected, rtol=1e-7, atol=1e-7)
    print("pass: basic asum")

    # strided
    x2 = torch.randn(200, dtype=torch.float64, device=device)
    result2 = torch.zeros(1, dtype=torch.float64, device=device)

    cublasDasum_v2(50, x2, 2, result2)
    expected2 = x2[::2][:50].abs().sum()
    torch.testing.assert_close(result2.squeeze(), expected2, rtol=1e-7, atol=1e-7)
    print("pass: strided asum")

    print("\ncublasDasum_v2 all tests passed")
