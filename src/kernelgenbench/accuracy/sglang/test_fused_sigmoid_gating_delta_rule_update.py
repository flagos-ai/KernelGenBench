"""
Accuracy and benchmark test for SGLang fused_sigmoid_gating_delta_rule_update.
Source: fused_sigmoid_gating_delta_rule_update(q,k,v,A_log,a,dt_bias,b,scale,initial_state,...)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("fused_sigmoid_gating_delta_rule_update")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_fused_sigmoid_gating_delta_rule_update(shape, dtype):
    M, N = shape
    B, H, K, HV, V = 1, 4, 64, 4, 64
    scale = K ** -0.5
    q = torch.randn(B, 1, H, K, device='cuda', dtype=dtype)
    k = torch.randn(B, 1, H, K, device='cuda', dtype=dtype)
    v = torch.randn(B, 1, HV, V, device='cuda', dtype=dtype)
    A_log = torch.randn(HV, device='cuda', dtype=dtype)
    a = torch.randn(B, 1, HV, device='cuda', dtype=dtype)
    dt_bias = torch.randn(HV, device='cuda', dtype=dtype)
    b = torch.randn(B, 1, HV, device='cuda', dtype=dtype)
    init_state = torch.randn(B, HV, V, K, device='cuda', dtype=dtype)
    ref_out = kernelgenbench.baseline.fused_sigmoid_gating_delta_rule_update(
        q=q, k=k, v=v, A_log=A_log, a=a, dt_bias=dt_bias, b=b, scale=scale,
        initial_state=init_state, cu_seqlens=None, initial_state_indices=None)
    act_out = kernelgenbench.triton.fused_sigmoid_gating_delta_rule_update(
        q=q.clone(), k=k.clone(), v=v.clone(), A_log=A_log, a=a.clone(), dt_bias=dt_bias, b=b.clone(), scale=scale,
        initial_state=init_state.clone(), cu_seqlens=None, initial_state_indices=None)
    assert_close(act_out, ref_out, dtype, strict=False)
    if M < 256:
        return None
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.fused_sigmoid_gating_delta_rule_update(q=q.clone(), k=k.clone(), v=v.clone(), A_log=A_log, a=a.clone(), dt_bias=dt_bias, b=b.clone(), scale=scale,
        initial_state=init_state.clone(), cu_seqlens=None, initial_state_indices=None),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.fused_sigmoid_gating_delta_rule_update(q=q.clone(), k=k.clone(), v=v.clone(), A_log=A_log, a=a.clone(), dt_bias=dt_bias, b=b.clone(), scale=scale,
        initial_state=init_state.clone(), cu_seqlens=None, initial_state_indices=None),
        warmup=25, rep=100
    )
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
