"""
Accuracy and benchmark test for SGLang mixer2_rms_norm_gated.
Source: Mixer2RMSNormGated(hidden_size, eps).forward_cuda(x, gate) -> Tensor
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("mixer2_rms_norm_gated")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_mixer2_rms_norm_gated(shape, dtype):
    M, N = shape
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    gate = torch.randn(M, N, device='cuda', dtype=dtype)
    ref_out = kernelgenbench.baseline.mixer2_rms_norm_gated(x, gate, hidden_size=N, eps=1e-6)
    act_out = kernelgenbench.triton.mixer2_rms_norm_gated(x.clone(), gate.clone(), hidden_size=N, eps=1e-6)
    assert_close(act_out, ref_out, dtype)
    if M < 256:
        return None
    x_b = torch.randn(M, N, device='cuda', dtype=dtype)
    g_b = torch.randn(M, N, device='cuda', dtype=dtype)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.mixer2_rms_norm_gated(x.clone(), gate.clone(), hidden_size=N, eps=1e-6),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.mixer2_rms_norm_gated(x.clone(), gate.clone(), hidden_size=N, eps=1e-6),
        warmup=25, rep=100
    )
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
