# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton


def _ceil_div(a, b):
    return (a + b - 1) // b


@label("moe_align_block_size")
@parametrize("num_tokens", [128, 1024, 4096, 6152, 11575, 16384])
@parametrize("num_experts", [4, 8, 16, 32, 64])
@parametrize("block_size", [8, 16, 32, 64])
@parametrize("topk", [2, 4, 10])
@parametrize("dtype", [torch.int32])
def test_accuracy_moe_align_block_size(num_tokens, num_experts, block_size, topk, dtype):
    torch.manual_seed(42)

    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), dtype=dtype, device=device)
    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)

    # ===== Baseline =====
    ref_sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtype, device=device)
    ref_expert_ids = torch.empty(max_num_tokens_padded // block_size, dtype=dtype, device=device)
    ref_num_tokens_post_pad = torch.empty(1, dtype=dtype, device=device)

    kernelgenbench.baseline.moe_align_block_size(
        topk_ids.clone(), num_experts, block_size,
        ref_sorted_ids, ref_expert_ids, ref_num_tokens_post_pad, None)

    # ===== Triton =====
    act_sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtype, device=device)
    act_expert_ids = torch.empty(max_num_tokens_padded // block_size, dtype=dtype, device=device)
    act_num_tokens_post_pad = torch.empty(1, dtype=dtype, device=device)

    kernelgenbench.triton.moe_align_block_size(
        topk_ids.clone(), num_experts, block_size,
        act_sorted_ids, act_expert_ids, act_num_tokens_post_pad, None)

    torch.cuda.synchronize()

    # ===== Accuracy: deterministic outputs =====
    assert_close(act_num_tokens_post_pad, ref_num_tokens_post_pad, dtype)
    assert_close(act_expert_ids, ref_expert_ids, dtype)

    # ===== Accuracy: sorted_ids semantic comparison =====
    # sorted_token_ids ordering within each expert is non-deterministic,
    # so compare token sets per expert instead of element-wise.
    total_tokens = topk_ids.numel()
    start = 0
    for eid in range(num_experts):
        cnt = (topk_ids == eid).sum().item()
        aligned = _ceil_div(cnt, block_size) * block_size
        end = start + aligned

        ref_set = set(ref_sorted_ids[start:end].tolist())
        act_set = set(act_sorted_ids[start:end].tolist())
        assert ref_set == act_set, (
            f"Expert {eid}: token sets differ.\n"
            f"  ref: {sorted(ref_set)}\n  act: {sorted(act_set)}")

        start = end

    # ===== Performance Test =====
    if num_tokens < 1024:
        return None

    def _bench_baseline():
        s = torch.empty(max_num_tokens_padded, dtype=dtype, device=device)
        e = torch.empty(max_num_tokens_padded // block_size, dtype=dtype, device=device)
        n = torch.empty(1, dtype=dtype, device=device)
        kernelgenbench.baseline.moe_align_block_size(
            topk_ids.clone(), num_experts, block_size, s, e, n, None)

    def _bench_triton():
        s = torch.empty(max_num_tokens_padded, dtype=dtype, device=device)
        e = torch.empty(max_num_tokens_padded // block_size, dtype=dtype, device=device)
        n = torch.empty(1, dtype=dtype, device=device)
        kernelgenbench.triton.moe_align_block_size(
            topk_ids.clone(), num_experts, block_size, s, e, n, None)

    ms_baseline = triton.testing.do_bench(_bench_baseline, warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(_bench_triton, warmup=25, rep=100)

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
