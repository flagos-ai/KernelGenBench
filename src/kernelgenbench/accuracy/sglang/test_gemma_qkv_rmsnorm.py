"""
Accuracy and benchmark test for SGLang gemma_qkv_rmsnorm.
Source: gemma_qkv_rmsnorm(q [N,QH,Hdim], k [N,KH,Hdim], v [N,KH,Hdim], q_weight, k_weight, num_q_heads, num_kv_heads, head_dim, eps) -> None (in-place)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("gemma_qkv_rmsnorm")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_gemma_qkv_rmsnorm(shape, dtype):
    M, N = shape
    num_qh, num_kh, hdim = 4, 2, N // 4
    q = torch.randn(M, num_qh * hdim, device='cuda', dtype=dtype)
    k = torch.randn(M, num_kh * hdim, device='cuda', dtype=dtype)
    v = torch.randn(M, num_kh * hdim, device='cuda', dtype=dtype)
    q_weight = torch.randn(hdim, device='cuda', dtype=dtype)
    k_weight = torch.randn(hdim, device='cuda', dtype=dtype)
    ref_q, ref_k, ref_v = q.clone(), k.clone(), v.clone()
    act_q, act_k, act_v = q.clone(), k.clone(), v.clone()
    kernelgenbench.baseline.gemma_qkv_rmsnorm(ref_q, ref_k, ref_v, q_weight, k_weight, num_q_heads=num_qh, num_kv_heads=num_kh, head_dim=hdim, eps=1e-6)
    kernelgenbench.triton.gemma_qkv_rmsnorm(act_q, act_k, act_v, q_weight, k_weight, num_q_heads=num_qh, num_kv_heads=num_kh, head_dim=hdim, eps=1e-6)
    assert_close(act_q, ref_q, dtype)
    assert_close(act_k, ref_k, dtype)
    if M < 256:
        return None
    q_b = torch.randn(M, num_qh * hdim, device='cuda', dtype=dtype)
    k_b = torch.randn(M, num_kh * hdim, device='cuda', dtype=dtype)
    v_b = torch.randn(M, num_kh * hdim, device='cuda', dtype=dtype)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.gemma_qkv_rmsnorm(q_b, k_b, v_b, q_weight, k_weight, num_q_heads=num_qh, num_kv_heads=num_kh, head_dim=hdim, eps=1e-6),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.gemma_qkv_rmsnorm(q_b.clone(), k_b.clone(), v_b.clone(), q_weight, k_weight, num_q_heads=num_qh, num_kv_heads=num_kh, head_dim=hdim, eps=1e-6),
        warmup=25, rep=100
    )
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
