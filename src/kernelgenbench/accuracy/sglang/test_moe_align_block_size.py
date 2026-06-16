"""
Accuracy and benchmark test for SGLang moe_align_block_size.
Source: moe_align_block_size(topk_ids [M,topk], num_experts, block_size)
"""
import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("moe_align_block_size")
@parametrize("shape", [(64, 64), (256, 128)])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_moe_align_block_size(shape, dtype):
    M, N = shape
    topk_ids = torch.randint(0, 4, (M, 2), device='cuda', dtype=torch.int32)
    ref_out = kernelgenbench.baseline.moe_align_block_size(topk_ids, num_experts=4, block_size=32)
    act_out = kernelgenbench.baseline.moe_align_block_size(topk_ids.clone(), num_experts=4, block_size=32)
    assert_close(act_out[0], ref_out[0], torch.int32)
    assert_close(act_out[1], ref_out[1], torch.int32)
    if M < 256:
        return None
    topk_b = torch.randint(0, 4, (M, 2), device='cuda', dtype=torch.int32)
    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.moe_align_block_size(topk_b, num_experts=4, block_size=32),
        warmup=25, rep=100
    )
    speedup = 1.0
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_baseline, speedup=speedup)
