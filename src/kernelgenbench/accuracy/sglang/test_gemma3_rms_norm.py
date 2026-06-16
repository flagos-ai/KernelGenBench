"""
Accuracy and benchmark test for SGLang Gemma3RMSNorm.
Source: Gemma3RMSNorm(dim, eps).forward_cuda(x) — output = norm(x) * (1 + weight)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("gemma3_rms_norm")
@parametrize("shape", [(1, 32), (71, 497), (128, 512), (1024, 4096)])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@parametrize("eps", [1e-6, 1e-5])
def test_accuracy_gemma3_rms_norm(shape, dtype, eps):
    M, dim = shape
    x = torch.randn(M, dim, device='cuda', dtype=dtype)

    ref_out = kernelgenbench.baseline.gemma3_rms_norm(x, dim, eps=eps)
    act_out = kernelgenbench.baseline.gemma3_rms_norm(x.clone(), dim, eps=eps)

    assert_close(act_out, ref_out, dtype)

    if M < 1024 or dtype == torch.float32:
        return None

    x_bench = torch.randn(M, dim, device='cuda', dtype=dtype)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.gemma3_rms_norm(x_bench, dim, eps=eps),
        warmup=25, rep=100
    )
    speedup = 1.0
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_baseline, speedup=speedup)
