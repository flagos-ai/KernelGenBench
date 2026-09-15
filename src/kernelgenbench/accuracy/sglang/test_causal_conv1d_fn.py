"""
Accuracy and benchmark test for SGLang causal_conv1d_fn.
Source: causal_conv1d_fn(x [B,T,D], weight [D,width], bias, query_start_loc, ...) (sgl_kernel)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("causal_conv1d_fn")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_causal_conv1d_fn(shape, dtype):
    M, N = shape
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    weight = torch.randn(N, 4, device='cuda', dtype=dtype)
    ref_out = kernelgenbench.baseline.causal_conv1d_fn(x, weight, bias=None, query_start_loc=None, cache_indices=None, has_initial_state=None, conv_states=None, activation='silu')
    act_out = kernelgenbench.triton.causal_conv1d_fn(x.clone(), weight, bias=None, query_start_loc=None, cache_indices=None, has_initial_state=None, conv_states=None, activation='silu')
    assert_close(act_out, ref_out, dtype)
    if M < 256:
        return None
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.causal_conv1d_fn(x.clone(), weight, bias=None, query_start_loc=None, cache_indices=None, has_initial_state=None, conv_states=None, activation='silu'),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.causal_conv1d_fn(x, weight, bias=None, query_start_loc=None, cache_indices=None, has_initial_state=None, conv_states=None, activation='silu'),
        warmup=25, rep=100
    )
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
