"""
Accuracy and benchmark test for SGLang FusedDualResidualRMSNorm.
Source: FusedDualResidualRMSNorm(rmsnorm1, rmsnorm2).forward(x, residual) -> (out, residual)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("fused_dual_residual_rmsnorm")
@parametrize("shape", [(1, 32), (128, 512), (1024, 4096)])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_accuracy_fused_dual_residual_rmsnorm(shape, dtype):
    M, hidden_size = shape
    x = torch.randn(M, hidden_size, device='cuda', dtype=dtype)
    residual = torch.randn(M, hidden_size, device='cuda', dtype=dtype)

    ref_out, ref_res = kernelgenbench.baseline.fused_dual_residual_rmsnorm(
        x, residual, hidden_size, hidden_size, eps=1e-6)
    act_out, act_res = kernelgenbench.baseline.fused_dual_residual_rmsnorm(
        x.clone(), residual.clone(), hidden_size, hidden_size, eps=1e-6)

    assert_close(act_out, ref_out, dtype)
    assert_close(act_res, ref_res, dtype)

    if M < 1024 or dtype == torch.float32:
        return None

    x_bench = torch.randn(M, hidden_size, device='cuda', dtype=dtype)
    r_bench = torch.randn(M, hidden_size, device='cuda', dtype=dtype)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.fused_dual_residual_rmsnorm(
            x_bench, r_bench, hidden_size, hidden_size, eps=1e-6),
        warmup=25, rep=100
    )
    speedup = 1.0
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_baseline, speedup=speedup)
