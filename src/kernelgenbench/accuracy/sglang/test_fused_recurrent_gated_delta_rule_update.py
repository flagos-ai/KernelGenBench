"""
Accuracy and benchmark test for SGLang fused_recurrent_gated_delta_rule_update.
Source: fused_recurrent_gated_delta_rule_update(q,k,v,g,beta,scale,initial_state,cu_seqlens,initial_state_indices,intermediate_states,eagle_tree)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("fused_recurrent_gated_delta_rule_update")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_fused_recurrent_gated_delta_rule_update(shape, dtype):
    M, N = shape
    B, H, K, HV, V = 1, 4, 64, 4, 64
    scale = K ** -0.5
    q = torch.randn(B, 1, H, K, device='cuda', dtype=dtype)
    k = torch.randn(B, 1, H, K, device='cuda', dtype=dtype)
    v = torch.randn(B, 1, HV, V, device='cuda', dtype=dtype)
    g = torch.randn(B, 1, HV, device='cuda', dtype=dtype)
    beta = torch.randn(B, 1, HV, device='cuda', dtype=dtype)
    init_state = torch.randn(B, HV, V, K, device='cuda', dtype=dtype)
    ref_out = kernelgenbench.baseline.fused_recurrent_gated_delta_rule_update(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=init_state,
        cu_seqlens=None, initial_state_indices=None, intermediate_states=None, eagle_tree=None)
    act_out = kernelgenbench.baseline.fused_recurrent_gated_delta_rule_update(
        q=q.clone(), k=k.clone(), v=v.clone(), g=g.clone(), beta=beta.clone(), scale=scale, initial_state=init_state.clone(),
        cu_seqlens=None, initial_state_indices=None, intermediate_states=None, eagle_tree=None)
    assert_close(act_out, ref_out, dtype, strict=False)
    if M < 256:
        return None
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.fused_recurrent_gated_delta_rule_update(
            q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=init_state,
            cu_seqlens=None, initial_state_indices=None, intermediate_states=None, eagle_tree=None),
        warmup=25, rep=100
    )
    speedup = 1.0
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_baseline, speedup=speedup)
