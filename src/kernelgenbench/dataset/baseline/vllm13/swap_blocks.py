"""
vLLM swap_blocks baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def swap_blocks(
    src: torch.Tensor,
    dst: torch.Tensor,
    block_mapping: torch.Tensor
) -> None:
    """Copy specific blocks from one tensor to another."""
    _custom_ops.swap_blocks(
        src,
        dst,
        block_mapping
    )
