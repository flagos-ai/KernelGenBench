"""
Accuracy and benchmark test for SGLang rotary_embedding.
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("rotary_embedding")
@parametrize("seq_len", [128, 512, 1024, 4096])
@parametrize("num_heads", [4, 16])
@parametrize("head_size", [64, 128])
@parametrize("dtype", [torch.float16, torch.bfloat16])
@parametrize("is_neox", [True, False])
def test_accuracy_rotary_embedding(seq_len, num_heads, head_size, dtype, is_neox):
    """Accuracy and performance test for SGLang rotary_embedding."""
    # ===== Accuracy Test =====
    positions = torch.randint(0, 8192, (seq_len,), device='cuda', dtype=torch.long)
    q0 = torch.randn(seq_len, num_heads, head_size, device='cuda', dtype=dtype)
    k0 = torch.randn(seq_len, num_heads, head_size, device='cuda', dtype=dtype)

    ref_q, ref_k = q0.clone(), k0.clone()
    act_q, act_k = q0.clone(), k0.clone()

    kernelgenbench.baseline.rotary_embedding(positions, ref_q, ref_k, is_neox_style=is_neox, head_size=head_size, rotary_dim=head_size)
    kernelgenbench.triton.rotary_embedding(positions, act_q, act_k, is_neox_style=is_neox, head_size=head_size, rotary_dim=head_size)

    assert_close(act_q, ref_q, dtype)
    assert_close(act_k, ref_k, dtype)

    # ===== Performance Test =====
    if seq_len < 1024:
        return None

    pos_bench = torch.randint(0, 8192, (seq_len,), device='cuda', dtype=torch.long)
    q_bench = torch.randn(seq_len, num_heads, head_size, device='cuda', dtype=dtype)
    k_bench = torch.randn(seq_len, num_heads, head_size, device='cuda', dtype=dtype)

    def bench_baseline():
        q, k = q_bench.clone(), k_bench.clone()
        kernelgenbench.baseline.rotary_embedding(pos_bench, q, k, is_neox_style=is_neox, head_size=head_size, rotary_dim=head_size)

    def bench_triton():
        q, k = q_bench.clone(), k_bench.clone()
        kernelgenbench.triton.rotary_embedding(pos_bench, q, k, is_neox_style=is_neox, head_size=head_size, rotary_dim=head_size)

    ms_baseline = triton.testing.do_bench(bench_baseline, warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(bench_triton, warmup=25, rep=100)
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
