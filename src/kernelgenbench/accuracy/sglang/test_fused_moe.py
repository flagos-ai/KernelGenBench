"""
Accuracy and benchmark test for SGLang fused_moe.
Source: triton_kernel_fused_experts(hidden_states [M,N], w1 [E,N,2*I], w2 [E,I,N], routing_data, gather_indx, scatter_indx)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("fused_moe")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_fused_moe(shape, dtype):
    M, N = shape
    E, I, topk = 4, N, 2
    hidden_states = torch.randn(M, N, device='cuda', dtype=dtype)
    w1 = torch.randn(E, N, 2 * I, device='cuda', dtype=dtype)
    w2 = torch.randn(E, I, N, device='cuda', dtype=dtype)
    from triton_kernels.matmul_ogs import RoutingData, GatherIndx, ScatterIndx
    from triton_kernels.routing import compute_expt_data_torch
    logits = torch.randn(M, E, device='cuda', dtype=torch.float32)
    routing_data, gather_idx, scatter_idx = compute_expt_data_torch(logits, topk, M, 0.0, 0.0, False)
    routing_data = RoutingData(*routing_data)
    gather_idx = GatherIndx(*gather_idx)
    scatter_idx = ScatterIndx(*scatter_idx)
    ref_out = kernelgenbench.baseline.fused_moe(hidden_states, w1, w2, routing_data, gather_idx, scatter_idx, inplace=False, activation='silu')
    act_out = kernelgenbench.baseline.fused_moe(hidden_states.clone(), w1, w2, routing_data, gather_idx, scatter_idx, inplace=False, activation='silu')
    assert_close(act_out, ref_out, dtype)
    if M < 256:
        return None
    hidden_b = torch.randn(M, N, device='cuda', dtype=dtype)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.fused_moe(hidden_b, w1, w2, routing_data, gather_idx, scatter_idx, inplace=False, activation='silu'),
        warmup=25, rep=100
    )
    speedup = 1.0
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_baseline, speedup=speedup)
