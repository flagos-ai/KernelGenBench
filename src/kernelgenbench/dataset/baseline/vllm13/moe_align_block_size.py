"""
vLLM moe_align_block_size baseline wrapper.

This wrapper calls the vLLM C++ implementation directly.
"""

try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    expert_map=None
):
    """
    Wrapper for vLLM's moe_align_block_size implementation.

    This is an in-place operation that modifies the output tensors directly.
    """
    _custom_ops.moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        expert_map
    )
