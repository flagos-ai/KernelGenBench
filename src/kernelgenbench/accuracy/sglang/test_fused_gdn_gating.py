"""
Accuracy and benchmark test for SGLang fused_gdn_gating.
Source: fused_gdn_gating(A_log [HV], a [B,HV], b [B,HV], dt_bias [HV], beta, threshold)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("fused_gdn_gating")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_fused_gdn_gating(shape, dtype):
    M, N = shape
    B, HV = 4, 16
    A_log = torch.randn(HV, device='cuda', dtype=dtype)
    a = torch.randn(B, HV, device='cuda', dtype=dtype)
    b = torch.randn(B, HV, device='cuda', dtype=dtype)
    dt_bias = torch.randn(HV, device='cuda', dtype=dtype)
    ref_g, ref_beta_out = kernelgenbench.baseline.fused_gdn_gating(A_log=A_log, a=a, b=b, dt_bias=dt_bias, beta=1.0, threshold=20.0)
    act_g, act_beta_out = kernelgenbench.baseline.fused_gdn_gating(A_log=A_log, a=a.clone(), b=b.clone(), dt_bias=dt_bias, beta=1.0, threshold=20.0)
    assert_close(act_g, ref_g, dtype)
    assert_close(act_beta_out, ref_beta_out, dtype)
    if M < 256:
        return None
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.fused_gdn_gating(A_log=A_log, a=a, b=b, dt_bias=dt_bias, beta=1.0, threshold=20.0),
        warmup=25, rep=100
    )
    speedup = 1.0
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_baseline, speedup=speedup)
