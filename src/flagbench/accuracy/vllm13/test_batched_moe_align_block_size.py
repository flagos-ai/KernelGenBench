import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import gems_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("batched_moe_align_block_size")
@parametrize("config", [
    (4, 4, 32),
    (8, 8, 64),
    (16, 4, 128),
    (8, 4, 256),
    (32, 8, 128),
    (32, 8, 64),
    (64, 8, 128),
    (64, 16, 256),
    (128, 16, 256),
])
def test_accuracy_batched_moe_align_block_size(config):
    num_experts = config[0]
    block_size = config[1]
    max_tokens = config[2]

    expert_num_tokens = torch.randint(1, max_tokens // 2,
                                      (num_experts,), device=device, dtype=torch.int32)
    padded_max = ((max_tokens + block_size - 1) // block_size) * block_size
    sorted_ids_size = num_experts * padded_max
    num_blocks = sorted_ids_size // block_size

    ref_sorted = torch.zeros(sorted_ids_size, device=device, dtype=torch.int32)
    ref_experts = torch.zeros(num_blocks, device=device, dtype=torch.int32)
    ref_ntp = torch.zeros(1, device=device, dtype=torch.int32)

    act_sorted = torch.zeros(sorted_ids_size, device=device, dtype=torch.int32)
    act_experts = torch.zeros(num_blocks, device=device, dtype=torch.int32)
    act_ntp = torch.zeros(1, device=device, dtype=torch.int32)

    flagbench.baseline.batched_moe_align_block_size(
        max_tokens, block_size, expert_num_tokens,
        ref_sorted, ref_experts, ref_ntp)

    flagbench.triton.batched_moe_align_block_size(
        max_tokens, block_size, expert_num_tokens,
        act_sorted, act_experts, act_ntp)

    assert torch.equal(act_ntp, ref_ntp), \
        f"num_tokens_post_pad mismatch: {act_ntp.item()} vs {ref_ntp.item()}"
    assert torch.equal(act_sorted, ref_sorted), \
        f"sorted_ids mismatch: max diff={(act_sorted - ref_sorted).abs().max()}"
    assert torch.equal(act_experts, ref_experts), \
        f"experts_ids mismatch: max diff={(act_experts - ref_experts).abs().max()}"

    # ===== Performance Test =====
    if num_experts < 16 or max_tokens < 128:
        return None

    expert_num_tokens_bench = torch.randint(1, max_tokens // 2,
                                            (num_experts,), device=device, dtype=torch.int32)
    sorted_baseline = torch.zeros(sorted_ids_size, device=device, dtype=torch.int32)
    experts_baseline = torch.zeros(num_blocks, device=device, dtype=torch.int32)
    ntp_baseline = torch.zeros(1, device=device, dtype=torch.int32)

    sorted_triton = torch.zeros(sorted_ids_size, device=device, dtype=torch.int32)
    experts_triton = torch.zeros(num_blocks, device=device, dtype=torch.int32)
    ntp_triton = torch.zeros(1, device=device, dtype=torch.int32)

    ms_baseline = triton.testing.do_bench(
        lambda: flagbench.baseline.batched_moe_align_block_size(
            max_tokens, block_size, expert_num_tokens_bench,
            sorted_baseline, experts_baseline, ntp_baseline),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: flagbench.triton.batched_moe_align_block_size(
            max_tokens, block_size, expert_num_tokens_bench,
            sorted_triton, experts_triton, ntp_triton),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
