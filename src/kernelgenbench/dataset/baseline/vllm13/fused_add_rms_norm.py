"""
vLLM fused_add_rms_norm baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float
) -> None:
    """Wrapper for vLLM fused_add_rms_norm implementation."""
    _custom_ops.fused_add_rms_norm(
        input,
        residual,
        weight,
        epsilon
    )
