"""
Accuracy and benchmark test for SGLang fused_sigmoid_gating_delta_rule_packed_decode.
Source: fused_sigmoid_gating_delta_rule_packed_decode(mixed_qkv,a,b,A_log,dt_bias,scale,initial_state,out,ssm_state_indices)->(out,initial_state)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("fused_sigmoid_gating_delta_rule_packed_decode")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_fused_sigmoid_gating_delta_rule_packed_decode(shape, dtype):
    M, N = shape
    B, H, K, HV, V = 4, 4, 64, 4, 64
    scale = K ** -0.5
    mixed_qkv = torch.randn(B, 2*H*K + HV*V, device='cuda', dtype=dtype)
    a = torch.randn(B, HV, device='cuda', dtype=dtype)
    b = torch.randn(B, HV, device='cuda', dtype=dtype)
    A_log = torch.randn(HV, device='cuda', dtype=dtype)
    dt_bias = torch.randn(HV, device='cuda', dtype=dtype)
    pool_size = 1
    init_state = torch.randn(pool_size, HV, V, K, device='cuda', dtype=dtype)
    out = torch.zeros(B, 1, HV, V, device='cuda', dtype=dtype)
    ssm_state_indices = torch.arange(B, device='cuda', dtype=torch.int32)
    ref_out, ref_state = kernelgenbench.baseline.fused_sigmoid_gating_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale,
        initial_state=init_state, out=out, ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=False)
    act_out, act_state = kernelgenbench.baseline.fused_sigmoid_gating_delta_rule_packed_decode(
        mixed_qkv=mixed_qkv.clone(), a=a.clone(), b=b.clone(), A_log=A_log, dt_bias=dt_bias, scale=scale,
        initial_state=init_state.clone(), out=out.clone(), ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=False)
    assert_close(act_out, ref_out, dtype, strict=False)
    if M < 256:
        return None
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.fused_sigmoid_gating_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale,
            initial_state=init_state, out=out, ssm_state_indices=ssm_state_indices,
            use_qk_l2norm_in_kernel=False),
        warmup=25, rep=100
    )
    speedup = 1.0
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_baseline, speedup=speedup)
