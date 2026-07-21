"""
Accuracy and benchmark test for SGLang fused_recurrent_gated_delta_rule.
Source: fused_recurrent_gated_delta_rule(q[B,T,H,K], k[B,T,H,K], v[B,T,HV,V], g[B,T,HV], beta[B,T,HV], scale)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("fused_recurrent_gated_delta_rule")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_fused_recurrent_gated_delta_rule(shape, dtype):
    M, N = shape
    B, T, H, K, HV, V = 1, 64, 4, 64, 4, 64
    scale = K ** -0.5
    q = torch.randn(B, T, H, K, device='cuda', dtype=dtype)
    k = torch.randn(B, T, H, K, device='cuda', dtype=dtype)
    v = torch.randn(B, T, HV, V, device='cuda', dtype=dtype)
    g = torch.randn(B, T, HV, device='cuda', dtype=dtype)
    beta = torch.randn(B, T, HV, device='cuda', dtype=dtype)
    ref_out, ref_state = kernelgenbench.baseline.fused_recurrent_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=None, output_final_state=False, cu_seqlens=None)
    act_out, act_state = kernelgenbench.triton.fused_recurrent_gated_delta_rule(
        q=q.clone(), k=k.clone(), v=v.clone(), g=g.clone(), beta=beta.clone(), scale=scale, initial_state=None, output_final_state=False, cu_seqlens=None)
    assert_close(act_out, ref_out, dtype)
    if M < 256:
        return None
    q_b = torch.randn(B, T, H, K, device='cuda', dtype=dtype)
    k_b = torch.randn(B, T, H, K, device='cuda', dtype=dtype)
    v_b = torch.randn(B, T, HV, V, device='cuda', dtype=dtype)
    g_b = torch.randn(B, T, HV, device='cuda', dtype=dtype)
    beta_b = torch.randn(B, T, HV, device='cuda', dtype=dtype)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.fused_recurrent_gated_delta_rule(q=q.clone(), k=k.clone(), v=v.clone(), g=g.clone(), beta=beta.clone(), scale=scale, initial_state=None, output_final_state=False, cu_seqlens=None),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.fused_recurrent_gated_delta_rule(q=q.clone(), k=k.clone(), v=v.clone(), g=g.clone(), beta=beta.clone(), scale=scale, initial_state=None, output_final_state=False, cu_seqlens=None),
        warmup=25, rep=100
    )
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
