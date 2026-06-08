"""
vLLM permute_cols baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def permute_cols(
    a: torch.Tensor,
    perm: torch.Tensor
) -> torch.Tensor:
    """Wrapper for vLLM permute_cols implementation."""
    return _custom_ops.permute_cols(
        a,
        perm
    )
