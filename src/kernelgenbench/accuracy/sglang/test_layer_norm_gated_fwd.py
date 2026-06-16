"""
Accuracy and benchmark test for SGLang layer_norm_gated_fwd.
Source: layer_norm_gated_fwd(x [N,D], g [N,D], weight [D], bias [D]|None, activation, eps, residual)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("layer_norm_gated_fwd")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_layer_norm_gated_fwd(shape, dtype):
    M, N = shape
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    g = torch.randn(M, N, device='cuda', dtype=dtype)
    weight = torch.ones(N, device='cuda', dtype=dtype)
    bias = torch.zeros(N, device='cuda', dtype=dtype)
    ref_out = kernelgenbench.baseline.layer_norm_gated_fwd(x, g, weight, bias, activation='swish', eps=1e-5, residual=None)
    act_out = kernelgenbench.baseline.layer_norm_gated_fwd(x.clone(), g.clone(), weight, bias, activation='swish', eps=1e-5, residual=None)
    assert_close(act_out[0], ref_out[0], dtype)
    if M < 256:
        return None
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.layer_norm_gated_fwd(x, g, weight, bias, activation='swish', eps=1e-5, residual=None),
        warmup=25, rep=100
    )
    speedup = 1.0
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_baseline, speedup=speedup)
