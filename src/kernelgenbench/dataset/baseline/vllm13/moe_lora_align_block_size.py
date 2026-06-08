"""vLLM moe_lora_align_block_size baseline wrapper."""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


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
    _custom_ops.moe_lora_align_block_size(
        topk_ids, token_lora_mapping, num_experts, block_size,
        max_loras, max_num_tokens_padded, max_num_m_blocks,
        sorted_token_ids, experts_ids, num_tokens_post_pad,
        adapter_enabled, lora_ids, expert_map
    )
