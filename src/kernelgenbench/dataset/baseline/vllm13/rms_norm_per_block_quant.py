"""
vLLM rms_norm_per_block_quant baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def rms_norm_per_block_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    group_size: list,
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    is_scale_transposed: bool = False
) -> tuple:
    """Wrapper for vLLM rms_norm_per_block_quant implementation."""
    return _custom_ops.rms_norm_per_block_quant(
        input,
        weight,
        epsilon,
        quant_dtype,
        group_size,
        scale_ub,
        residual,
        is_scale_transposed
    )
