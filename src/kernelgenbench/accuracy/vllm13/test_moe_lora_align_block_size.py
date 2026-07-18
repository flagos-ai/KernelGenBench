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

@label("moe_lora_align_block_size")
@parametrize("config", [
    (16, 2, 8, 4, 2),
    (32, 2, 8, 8, 4),
    (64, 4, 16, 4, 2),
    (128, 2, 16, 8, 4),
    (128, 4, 16, 4, 2),
    (32, 4, 8, 4, 2),
    (256, 4, 16, 8, 4),
    (256, 2, 16, 4, 2),
    (128, 2, 32, 8, 4),
    (64, 2, 8, 4, 2),
])
def test_accuracy_moe_lora_align_block_size(config):
    num_tokens = config[0]
    topk = config[1]
    num_experts = config[2]
    block_size = config[3]
    max_loras = config[4]

    max_num_tokens_padded = num_tokens * topk + num_experts * block_size
    max_num_m_blocks = max_num_tokens_padded // block_size

    topk_ids = torch.randint(0, num_experts, (num_tokens, topk),
                             device=device, dtype=torch.int32)
    token_lora_mapping = torch.zeros(num_tokens, device=device, dtype=torch.int32)

    ref_sorted = torch.zeros(max_num_tokens_padded, device=device, dtype=torch.int32)
    ref_experts = torch.zeros(max_num_m_blocks, device=device, dtype=torch.int32)
    ref_ntp = torch.zeros(1, device=device, dtype=torch.int32)
    ref_adapter = torch.zeros(max_num_m_blocks, device=device, dtype=torch.int32)
    ref_lora = torch.zeros(max_num_m_blocks, device=device, dtype=torch.int32)

    act_sorted = torch.zeros(max_num_tokens_padded, device=device, dtype=torch.int32)
    act_experts = torch.zeros(max_num_m_blocks, device=device, dtype=torch.int32)
    act_ntp = torch.zeros(1, device=device, dtype=torch.int32)
    act_adapter = torch.zeros(max_num_m_blocks, device=device, dtype=torch.int32)
    act_lora = torch.zeros(max_num_m_blocks, device=device, dtype=torch.int32)

    kernelgenbench.baseline.moe_lora_align_block_size(
        topk_ids, token_lora_mapping, num_experts, block_size,
        max_loras, max_num_tokens_padded, max_num_m_blocks,
        ref_sorted, ref_experts, ref_ntp, ref_adapter, ref_lora, None)

    kernelgenbench.triton.moe_lora_align_block_size(
        topk_ids, token_lora_mapping, num_experts, block_size,
        max_loras, max_num_tokens_padded, max_num_m_blocks,
        act_sorted, act_experts, act_ntp, act_adapter, act_lora, None)

    assert torch.equal(act_sorted, ref_sorted), \
        f"sorted_token_ids mismatch: max diff={(act_sorted - ref_sorted).abs().max()}"
    assert torch.equal(act_experts, ref_experts), \
        f"experts_ids mismatch: max diff={(act_experts - ref_experts).abs().max()}"
    assert torch.equal(act_ntp, ref_ntp), \
        f"num_tokens_post_pad mismatch: {act_ntp.item()} vs {ref_ntp.item()}"

    # ===== Performance Test =====
    if num_tokens < 128 or num_experts < 16:
        return None

    topk_ids_bench = torch.randint(0, num_experts, (num_tokens, topk),
                                   device=device, dtype=torch.int32)
    token_lora_mapping_bench = torch.zeros(num_tokens, device=device, dtype=torch.int32)
    sorted_baseline = torch.zeros(max_num_tokens_padded, device=device, dtype=torch.int32)
    experts_baseline = torch.zeros(max_num_m_blocks, device=device, dtype=torch.int32)
    ntp_baseline = torch.zeros(1, device=device, dtype=torch.int32)
    adapter_baseline = torch.zeros(max_num_m_blocks, device=device, dtype=torch.int32)
    lora_baseline = torch.zeros(max_num_m_blocks, device=device, dtype=torch.int32)
    sorted_triton = torch.zeros(max_num_tokens_padded, device=device, dtype=torch.int32)
    experts_triton = torch.zeros(max_num_m_blocks, device=device, dtype=torch.int32)
    ntp_triton = torch.zeros(1, device=device, dtype=torch.int32)
    adapter_triton = torch.zeros(max_num_m_blocks, device=device, dtype=torch.int32)
    lora_triton = torch.zeros(max_num_m_blocks, device=device, dtype=torch.int32)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.moe_lora_align_block_size(
            topk_ids_bench, token_lora_mapping_bench, num_experts, block_size,
            max_loras, max_num_tokens_padded, max_num_m_blocks,
            sorted_baseline, experts_baseline, ntp_baseline,
            adapter_baseline, lora_baseline, None),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.moe_lora_align_block_size(
            topk_ids_bench, token_lora_mapping_bench, num_experts, block_size,
            max_loras, max_num_tokens_padded, max_num_m_blocks,
            sorted_triton, experts_triton, ntp_triton,
            adapter_triton, lora_triton, None),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
