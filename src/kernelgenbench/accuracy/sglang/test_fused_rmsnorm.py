"""
Accuracy and benchmark test for SGLang fused_rmsnorm.
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("fused_rmsnorm")
@parametrize("shape", [(1, 512), (71, 2048), (128, 4096), (1024, 8192), (5333, 4096)])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_accuracy_fused_rmsnorm(shape, dtype):
    """Accuracy and performance test for SGLang fused_rmsnorm."""
    # ===== Accuracy Test =====
    M, N = shape
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    x_act = x.clone()

    ref_out = kernelgenbench.baseline.fused_rmsnorm(x)
    act_out = kernelgenbench.triton.fused_rmsnorm(x_act)

    assert_close(act_out, ref_out, dtype)

    # ===== Performance Test =====
    if M < 1024 or N < 4096 or dtype == torch.float32:
        return None

    x_bench = torch.randn(M, N, device='cuda', dtype=dtype)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.fused_rmsnorm(x_bench),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.fused_rmsnorm(x_bench.clone()),
        warmup=25, rep=100
    )
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
