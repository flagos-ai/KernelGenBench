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

"""
Triton kernel implementation for moe_lora_align_block_size.

This function aligns tokens and experts into block-sized chunks for LoRA-based
Mixture-of-Experts (MoE) execution.
"""
import torch
import triton
import triton.language as tl


def ceil_div(a, b):
    return (a + b - 1) // b


@triton.jit
def _moe_lora_align_stage1_count(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread,
    topk: tl.constexpr,
):
    """
    Stage 1: Count tokens per (lora_id, expert_id) pair.
    Each program handles a portion of tokens.
    """
    pid = tl.program_id(0)
    if pid >= num_experts:
        return

    start_idx = pid * tokens_per_thread

    for i in range(tokens_per_thread):
        token_idx = start_idx + i
        if token_idx < numel:
            expert_id = tl.load(topk_ids_ptr + token_idx)
            if expert_id == pid:
                lora_id = tl.load(token_lora_mapping_ptr + token_idx // topk)
                if lora_id < 2:  # max_loras
                    off = lora_id * num_experts + expert_id
                    cnt = tl.load(tokens_cnts_ptr + off)
                    tl.store(tokens_cnts_ptr + off, cnt + 1)


@triton.jit
def _moe_lora_align_stage2_prefix_sum(
    tokens_cnts_ptr,
    num_experts: tl.constexpr,
    max_loras: tl.constexpr,
):
    """
    Stage 2: Compute prefix sum within each lora row.
    """
    pid = tl.program_id(0)
    if pid >= max_loras * num_experts:
        return

    lora_id = pid // num_experts
    expert_id = pid % num_experts

    if expert_id > 0:
        prev = tl.load(tokens_cnts_ptr + lora_id * num_experts + expert_id - 1)
        curr = tl.load(tokens_cnts_ptr + lora_id * num_experts + expert_id)
        tl.store(tokens_cnts_ptr + lora_id * num_experts + expert_id, curr + prev)


@triton.jit
def _moe_lora_align_stage3_padded_cumsum(
    tokens_cnts_ptr,
    cumsum_ptr,
    num_tokens_post_pad_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    max_loras: tl.constexpr,
):
    """
    Stage 3: Compute padded cumulative sum and total tokens.
    """
    pid = tl.program_id(0)
    if pid >= max_loras:
        return

    # Compute original counts from cumulative
    last_cnt = 0
    for expert_id in range(num_experts):
        if expert_id == 0:
            curr_cnt = tl.load(tokens_cnts_ptr + pid * num_experts)
        else:
            prev = tl.load(tokens_cnts_ptr + pid * num_experts + expert_id - 1)
            curr = tl.load(tokens_cnts_ptr + pid * num_experts + expert_id)
            curr_cnt = curr - prev

        last_cnt = last_cnt + tl.cdiv(curr_cnt, block_size) * block_size
        tl.store(cumsum_ptr + pid * (num_experts + 1) + expert_id + 1, last_cnt)

    tl.store(num_tokens_post_pad_ptr + pid, last_cnt)


@triton.jit
def _moe_lora_align_stage4_fill_expert_ids(
    cumsum_ptr,
    expert_ids_ptr,
    num_experts: tl.constexpr,
    block_size: tl.constexpr,
    max_loras: tl.constexpr,
    max_num_m_blocks: tl.constexpr,
):
    """
    Stage 4: Fill expert_ids for blocks.
    """
    pid = tl.program_id(0)
    if pid >= max_loras * num_experts:
        return

    lora_id = pid // num_experts
    expert_id = pid % num_experts

    start_idx = tl.load(cumsum_ptr + lora_id * (num_experts + 1) + expert_id)
    end_idx = tl.load(cumsum_ptr + lora_id * (num_experts + 1) + expert_id + 1)

    # Fill blocks for this expert
    base_block = start_idx // block_size
    num_blocks = (end_idx - start_idx + block_size - 1) // block_size

    for i in range(num_blocks):
        block_id = base_block + i
        if block_id < max_num_m_blocks:
            tl.store(expert_ids_ptr + block_id, expert_id)


@triton.jit
def _moe_lora_align_stage5_sort_tokens(
    topk_ids_ptr,
    token_lora_mapping_ptr,
    sorted_token_ids_ptr,
    tokens_cnts_ptr,
    cumsum_ptr,
    num_experts: tl.constexpr,
    numel,
    tokens_per_thread,
    topk: tl.constexpr,
    max_loras: tl.constexpr,
):
    """
    Stage 5: Sort tokens by (lora_id, expert_id).
    """
    pid = tl.program_id(0)
    if pid >= max_loras * num_experts:
        return

    lora_id = pid // num_experts
    expert_id = pid % num_experts

    # Reset count for this (lora_id, expert_id) to track placement
    # We need to know the original count, which is in cumsum
    start_offset = tl.load(cumsum_ptr + lora_id * (num_experts + 1) + expert_id)

    off_sorted = lora_id * num_experts * topk + start_offset

    # Process tokens and place them
    for i in range(tokens_per_thread):
        token_idx = i
        if token_idx < numel:
            current_expert = tl.load(topk_ids_ptr + token_idx)
            current_lora = tl.load(token_lora_mapping_ptr + token_idx // topk)

            if current_lora == lora_id and current_expert == expert_id:
                # Get current position
                off = lora_id * num_experts + expert_id
                cnt = tl.load(tokens_cnts_ptr + off)
                rank_post_pad = start_offset + cnt

                tl.store(sorted_token_ids_ptr + rank_post_pad, token_idx)
                tl.store(tokens_cnts_ptr + off, cnt + 1)


def moe_lora_align_block_size(
    topk_ids: torch.Tensor,
    token_lora_mapping: torch.Tensor,
    num_experts: int,
    block_size: int,
    max_loras: int,
    max_num_tokens_padded: int,
    max_num_m_blocks: int,
    sorted_token_ids: torch.Tensor,
    experts_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
    adapter_enabled: torch.Tensor,
    lora_ids: torch.Tensor,
    expert_map: "torch.Tensor | None" = None
) -> None:
    """
    Triton implementation of moe_lora_align_block_size.
    """
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1] if len(topk_ids.shape) > 1 else 1
    numel = num_tokens * topk

    # Flatten topk_ids to 1D
    topk_ids_flat = topk_ids.reshape(-1)

    tokens_per_thread = max(1, ceil_div(numel, num_experts))

    # Allocate temporary storage
    tokens_cnts = torch.zeros(
        (max_loras, num_experts), dtype=torch.int32, device=topk_ids.device
    )
    cumsum = torch.zeros(
        (max_loras * (num_experts + 1),), dtype=torch.int32, device=topk_ids.device
    )

    # Stage 1: Count tokens
    grid = (num_experts,)
    _moe_lora_align_stage1_count[grid](
        topk_ids_flat,
        token_lora_mapping,
        tokens_cnts,
        num_experts,
        numel,
        tokens_per_thread,
        topk,
    )

    # Stage 2: Prefix sum
    grid = (max_loras * num_experts,)
    _moe_lora_align_stage2_prefix_sum[grid](
        tokens_cnts,
        num_experts,
        max_loras,
    )

    # Stage 3: Padded cumsum
    grid = (max_loras,)
    _moe_lora_align_stage3_padded_cumsum[grid](
        tokens_cnts,
        cumsum,
        num_tokens_post_pad,
        num_experts,
        block_size,
        max_loras,
    )

    # Reset counts for final placement
    tokens_cnts.zero_()

    # Stage 4: Fill expert_ids
    grid = (max_loras * num_experts,)
    _moe_lora_align_stage4_fill_expert_ids[grid](
        cumsum,
        experts_ids,
        num_experts,
        block_size,
        max_loras,
        max_num_m_blocks,
    )

    # Stage 5: Sort tokens
    grid = (max_loras * num_experts,)
    _moe_lora_align_stage5_sort_tokens[grid](
        topk_ids_flat,
        token_lora_mapping,
        sorted_token_ids,
        tokens_cnts,
        cumsum,
        num_experts,
        numel,
        tokens_per_thread,
        topk,
        max_loras,
    )

    # Apply expert_map if provided
    if expert_map is not None:
        experts_ids[:] = expert_map[experts_ids]


def setup(
    num_tokens: int = 1024,
    num_experts: int = 8,
    block_size: int = 32,
    max_loras: int = 2,
    topk: int = 2,
    seed: int = None,
    **kwargs
):
    """
    Generate test data for moe_lora_align_block_size kernel.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Generate topk_ids: (num_tokens, topk) - indices of top-k experts for each token
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), device='cuda', dtype=torch.int32)

    # Generate token_lora_mapping: (num_tokens,) - lora adapter id for each token
    token_lora_mapping = torch.randint(0, max_loras, (num_tokens,), device='cuda', dtype=torch.int32)

    # Calculate max dimensions
    max_num_tokens_padded = num_tokens * topk
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    # Pre-allocate output tensors
    sorted_token_ids = torch.zeros(max_num_tokens_padded, device='cuda', dtype=torch.int32)
    experts_ids = torch.zeros(max_num_m_blocks, device='cuda', dtype=torch.int32)
    num_tokens_post_pad = torch.zeros(max_loras, device='cuda', dtype=torch.int32)
    adapter_enabled = torch.ones(max_loras, device='cuda', dtype=torch.int32)
    lora_ids = torch.arange(max_loras, device='cuda', dtype=torch.int32)

    return {
        "inputs": {
            "topk_ids": topk_ids,
            "token_lora_mapping": token_lora_mapping,
            "num_experts": num_experts,
            "block_size": block_size,
            "max_loras": max_loras,
            "max_num_tokens_padded": max_num_tokens_padded,
            "max_num_m_blocks": max_num_m_blocks,
            "sorted_token_ids": sorted_token_ids,
            "experts_ids": experts_ids,
            "num_tokens_post_pad": num_tokens_post_pad,
            "adapter_enabled": adapter_enabled,
            "lora_ids": lora_ids,
        },
        "outputs": ["sorted_token_ids", "experts_ids", "num_tokens_post_pad"]
    }


def run_kernel(**inputs):
    """
    Run the moe_lora_align_block_size kernel.
    """
    # Extract inputs
    topk_ids = inputs["topk_ids"]
    token_lora_mapping = inputs["token_lora_mapping"]
    num_experts = inputs["num_experts"]
    block_size = inputs["block_size"]
    max_loras = inputs["max_loras"]
    max_num_tokens_padded = inputs["max_num_tokens_padded"]
    max_num_m_blocks = inputs["max_num_m_blocks"]
    sorted_token_ids = inputs["sorted_token_ids"]
    experts_ids = inputs["experts_ids"]
    num_tokens_post_pad = inputs["num_tokens_post_pad"]
    adapter_enabled = inputs["adapter_enabled"]
    lora_ids = inputs["lora_ids"]

    # Call the main kernel
    moe_lora_align_block_size(
        topk_ids=topk_ids,
        token_lora_mapping=token_lora_mapping,
        num_experts=num_experts,
        block_size=block_size,
        max_loras=max_loras,
        max_num_tokens_padded=max_num_tokens_padded,
        max_num_m_blocks=max_num_m_blocks,
        sorted_token_ids=sorted_token_ids,
        experts_ids=experts_ids,
        num_tokens_post_pad=num_tokens_post_pad,
        adapter_enabled=adapter_enabled,
        lora_ids=lora_ids,
    )

    return sorted_token_ids, experts_ids, num_tokens_post_pad
