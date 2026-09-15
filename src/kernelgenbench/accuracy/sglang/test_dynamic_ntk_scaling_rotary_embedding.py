"""
Accuracy and benchmark test for SGLang dynamic_ntk_scaling_rotary_embedding.
Source: DynamicNTKScalingRotaryEmbedding(head_size,rotary_dim,max_positions,base,is_neox,dtype,scaling_factor).forward_cuda(pos,q,k)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("dynamic_ntk_scaling_rotary_embedding")
@parametrize("seq_len", [128, 512, 1024])
@parametrize("num_heads", [4, 16])
@parametrize("head_size", [64, 128])
@parametrize("dtype", [torch.float16, torch.bfloat16])
@parametrize("is_neox", [True, False])
def test_accuracy_dynamic_ntk_scaling_rotary_embedding(seq_len, num_heads, head_size, dtype, is_neox):
    positions = torch.arange(seq_len, device='cuda', dtype=torch.long)
    q0 = torch.randn(seq_len, num_heads, head_size, device='cuda', dtype=dtype)
    k0 = torch.randn(seq_len, num_heads, head_size, device='cuda', dtype=dtype)
    ref_q, ref_k = q0.clone(), k0.clone()
    act_q, act_k = q0.clone(), k0.clone()

    kernelgenbench.baseline.dynamic_ntk_scaling_rotary_embedding(ref_q, ref_k, positions, is_neox_style=is_neox, head_size=head_size, rotary_dim=head_size, scaling_factor=2.0)
    kernelgenbench.triton.dynamic_ntk_scaling_rotary_embedding(act_q, act_k, positions, is_neox_style=is_neox, head_size=head_size, rotary_dim=head_size, scaling_factor=2.0)
    assert_close(act_q, ref_q, dtype)
    assert_close(act_k, ref_k, dtype)
    if seq_len < 512:
        return None
    pos_b = torch.arange(seq_len, device='cuda', dtype=torch.long)
    q_b = torch.randn(seq_len, num_heads, head_size, device='cuda', dtype=dtype)
    k_b = torch.randn(seq_len, num_heads, head_size, device='cuda', dtype=dtype)
    def bench_baseline():
        q, k = q_b.clone(), k_b.clone()
        kernelgenbench.baseline.dynamic_ntk_scaling_rotary_embedding(q, k, pos_b, is_neox_style=is_neox, head_size=head_size, rotary_dim=head_size, scaling_factor=2.0)
    def bench_triton():
        q, k = q_b.clone(), k_b.clone()
        kernelgenbench.triton.dynamic_ntk_scaling_rotary_embedding(q, k, pos_b, is_neox_style=is_neox, head_size=head_size, rotary_dim=head_size, scaling_factor=2.0)
    ms_baseline = triton.testing.do_bench(bench_baseline, warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(bench_triton, warmup=25, rep=100)
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
