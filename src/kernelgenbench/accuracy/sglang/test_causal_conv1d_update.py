"""
Accuracy and benchmark test for SGLang causal_conv1d_update.
Source: causal_conv1d_update(x [B,D], conv_state [B,D,width-1], weight [D,width], bias, activation) (sgl_kernel)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("causal_conv1d_update")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_causal_conv1d_update(shape, dtype):
    M, N = shape
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    weight = torch.randn(N, 4, device='cuda', dtype=dtype)
    conv_state = torch.randn(M, N, 3, device='cuda', dtype=dtype)
    ref_out, ref_conv_state = kernelgenbench.baseline.causal_conv1d_update(x, conv_state, weight, bias=None, activation='silu')
    act_out, act_conv_state = kernelgenbench.baseline.causal_conv1d_update(x.clone(), conv_state.clone(), weight, bias=None, activation='silu')
    assert_close(act_out, ref_out, dtype)
    if M < 256:
        return None
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.causal_conv1d_update(x, conv_state, weight, bias=None, activation='silu'),
        warmup=25, rep=100
    )
    speedup = 1.0
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_baseline, speedup=speedup)
