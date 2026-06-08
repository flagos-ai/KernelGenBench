"""
vLLM convert_vertical_slash_indexes baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def convert_vertical_slash_indexes(
    q_seqlens: torch.Tensor,
    kv_seqlens: torch.Tensor,
    vertical_indexes: torch.Tensor,
    slash_indexes: torch.Tensor,
    context_size: int,
    block_size_M: int,
    block_size_N: int,
    causal: bool = True
) -> tuple:
    """Wrapper for vLLM convert_vertical_slash_indexes implementation."""
    return _custom_ops.convert_vertical_slash_indexes(
        q_seqlens,
        kv_seqlens,
        vertical_indexes,
        slash_indexes,
        context_size,
        block_size_M,
        block_size_N,
        causal
    )
