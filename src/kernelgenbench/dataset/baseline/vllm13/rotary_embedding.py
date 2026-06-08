"""
vLLM rotary_embedding baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool
) -> None:
    """Wrapper for vLLM rotary_embedding implementation."""
    _custom_ops.rotary_embedding(
        positions,
        query,
        key,
        head_size,
        cos_sin_cache,
        is_neox
    )
