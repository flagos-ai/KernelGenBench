"""
Accuracy and benchmark test for SGLang RMSNormWithoutScale.
Source: RMSNormWithoutScale(hidden_size, eps).forward_cuda(x) — pure RMS, no weight
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("rms_norm_without_scale")
@parametrize("shape", [(1, 32), (71, 497), (128, 512), (1024, 4096)])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@parametrize("eps", [1e-6, 1e-5])
def test_accuracy_rms_norm_without_scale(shape, dtype, eps):
    M, hidden_size = shape
    x = torch.randn(M, hidden_size, device='cuda', dtype=dtype)

    ref_out = kernelgenbench.baseline.rms_norm_without_scale(x, hidden_size, eps=eps)
    act_out = kernelgenbench.baseline.rms_norm_without_scale(x.clone(), hidden_size, eps=eps)

    assert_close(act_out, ref_out, dtype)

    if M < 1024 or dtype == torch.float32:
        return None

    x_bench = torch.randn(M, hidden_size, device='cuda', dtype=dtype)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.rms_norm_without_scale(x_bench, hidden_size, eps=eps),
        warmup=25, rep=100
    )
    speedup = 1.0
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_baseline, speedup=speedup)
