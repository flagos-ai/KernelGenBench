"""
Accuracy and benchmark test for SGLang topk.
Source: TopK(topk, renormalize).forward_cuda(hidden_states, router_logits) -> TopKOutput
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("topk")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_topk(shape, dtype):
    M, N = shape
    E = 8
    hidden_states = torch.randn(M, N, device='cuda', dtype=dtype)
    router_logits = torch.randn(M, E, device='cuda', dtype=torch.float32)
    ref_out = kernelgenbench.baseline.topk(hidden_states, router_logits, topk=2)
    act_out = kernelgenbench.triton.topk(hidden_states.clone(), router_logits.clone(), topk=2)
    assert_close(act_out.weights, ref_out.weights, torch.float32)
    if M < 256:
        return None
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.topk(hidden_states.clone(), router_logits.clone(), topk=2),
        warmup=25, rep=100
    )
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.topk(hidden_states.clone(), router_logits.clone(), topk=2),
        warmup=25, rep=100
    )
    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
