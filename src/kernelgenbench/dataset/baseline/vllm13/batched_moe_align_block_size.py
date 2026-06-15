"""vLLM batched_moe_align_block_size baseline wrapper."""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def batched_moe_align_block_size(
    max_tokens_per_batch: int,
    block_size: int,
    expert_num_tokens: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_pad: torch.Tensor,
) -> None:
    _custom_ops.batched_moe_align_block_size(
        max_tokens_per_batch, block_size, expert_num_tokens,
        sorted_ids, expert_ids, num_tokens_post_pad
    )
