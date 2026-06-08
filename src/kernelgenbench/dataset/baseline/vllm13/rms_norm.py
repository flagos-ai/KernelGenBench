"""
vLLM rms_norm baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def rms_norm(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float
) -> None:
    """Wrapper for vLLM rms_norm implementation."""
    _custom_ops.rms_norm(
        out,
        input,
        weight,
        epsilon
    )
