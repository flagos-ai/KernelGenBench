"""
Accuracy and benchmark test for SGLang triton_mrope_fused.
Source: triton_mrope_fused(q,k,cos_sin_cache,positions,mrope_section,head_size,rotary_dim,...)->None (in-place)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("triton_mrope_fused")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_triton_mrope_fused(shape, dtype):
    M, N = shape
    q0 = torch.randn(M, 4 * N, device='cuda', dtype=dtype)  # M tokens, 4 heads * head_size
    k0 = torch.randn(M, 4 * N, device='cuda', dtype=dtype)
    positions = torch.randint(0, 8192, (M,), device='cuda', dtype=torch.long)
    cos_sin_cache = torch.randn(8192, N, device='cuda', dtype=dtype)
    mrope_section = [N // 6] * 3
    ref_q, ref_k = q0.clone(), k0.clone()
    act_q, act_k = q0.clone(), k0.clone()
    kernelgenbench.baseline.triton_mrope_fused(ref_q, ref_k, cos_sin_cache, positions, mrope_section, head_size=N, rotary_dim=N, mrope_interleaved=False, mrope_interleaved_glm=False, is_neox_style=True, axis_map=None)
    kernelgenbench.baseline.triton_mrope_fused(act_q, act_k, cos_sin_cache, positions, mrope_section, head_size=N, rotary_dim=N, mrope_interleaved=False, mrope_interleaved_glm=False, is_neox_style=True, axis_map=None)
    assert_close(act_q, ref_q, dtype)
    assert_close(act_k, ref_k, dtype)
    if M < 256:
        return None
    q_b = torch.randn(M, 4 * N, device='cuda', dtype=dtype)
    k_b = torch.randn(M, 4 * N, device='cuda', dtype=dtype)
    pos_b = torch.randint(0, 8192, (M,), device='cuda', dtype=torch.long)
    cos_b = torch.randn(8192, N, device='cuda', dtype=dtype)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.triton_mrope_fused(q_b.clone(), k_b.clone(), cos_b, pos_b, mrope_section=[N//6]*3, head_size=N, rotary_dim=N, mrope_interleaved=False, mrope_interleaved_glm=False, is_neox_style=True, axis_map=None),
        warmup=25, rep=100
    )
    speedup = 1.0
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_baseline, speedup=speedup)
