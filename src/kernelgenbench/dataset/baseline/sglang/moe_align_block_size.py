"""
SGLang moe_align_block_size baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import moe_align_block_size as _moe_align
except ModuleNotFoundError:
    _moe_align = None



def moe_align_block_size(topk_ids: torch.Tensor, num_experts: int, block_size: int) -> "typing.Any":
    """
    Args:
        topk_ids: torch.Tensor
        num_experts: int
        block_size: int
    Returns:
        "typing.Any"
    """
    return _moe_align(topk_ids, num_experts, block_size)

