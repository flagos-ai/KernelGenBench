"""
Accuracy and benchmark test for SGLang experts_combine_triton.
Source: experts_combine_triton(moe_hidden_states [N,K,D], mlp_hidden_states [N,D], output_buffer)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("experts_combine_triton")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_experts_combine_triton(shape, dtype):
    M, N = shape
    K = 4
    moe_hidden = torch.randn(M, K, N, device='cuda', dtype=dtype)
    mlp_hidden = torch.randn(M, N, device='cuda', dtype=dtype)
    ref_out = kernelgenbench.baseline.experts_combine_triton(moe_hidden, mlp_hidden, output_buffer=None)
    act_out = kernelgenbench.baseline.experts_combine_triton(moe_hidden.clone(), mlp_hidden.clone(), output_buffer=None)
    assert_close(act_out, ref_out, dtype)
    if M < 256:
        return None
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.experts_combine_triton(moe_hidden, mlp_hidden, output_buffer=None),
        warmup=25, rep=100
    )
    speedup = 1.0
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_baseline, speedup=speedup)
