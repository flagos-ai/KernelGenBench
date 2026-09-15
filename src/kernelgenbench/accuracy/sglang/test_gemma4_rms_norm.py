"""
Accuracy and benchmark test for SGLang Gemma4RMSNorm.
Source: Gemma4RMSNorm(dim, eps, scale_shift).forward_cuda(x)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("gemma4_rms_norm")
@parametrize("shape", [(1, 32), (128, 512), (1024, 4096)])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@parametrize("scale_shift", [0.0, 1.0])
def test_accuracy_gemma4_rms_norm(shape, dtype, scale_shift):
    M, dim = shape
    x = torch.randn(M, dim, device='cuda', dtype=dtype)

    ref_out = kernelgenbench.baseline.gemma4_rms_norm(x, dim, eps=1e-6, scale_shift=scale_shift)
    act_out = kernelgenbench.triton.gemma4_rms_norm(x.clone(), dim, eps=1e-6, scale_shift=scale_shift)

    assert_close(act_out, ref_out, dtype)

    if M < 1024 or dtype == torch.float32:
        return None

    x_bench = torch.randn(M, dim, device='cuda', dtype=dtype)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.gemma4_rms_norm(x.clone(), dim, eps=1e-6, scale_shift=scale_shift),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.gemma4_rms_norm(x.clone(), dim, eps=1e-6, scale_shift=scale_shift),
        warmup=25, rep=100
    )
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
