"""
vLLM rms_norm_dynamic_per_token_quant baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def rms_norm_dynamic_per_token_quant(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    quant_dtype: torch.dtype,
    scale_ub: torch.Tensor | None = None,
    residual: torch.Tensor | None = None
) -> tuple:
    """Wrapper for vLLM rms_norm_dynamic_per_token_quant implementation."""
    return _custom_ops.rms_norm_dynamic_per_token_quant(
        input,
        weight,
        epsilon,
        quant_dtype,
        scale_ub,
        residual
    )
