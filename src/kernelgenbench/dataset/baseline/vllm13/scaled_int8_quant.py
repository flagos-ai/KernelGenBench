"""
vLLM scaled_int8_quant baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def scaled_int8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
    symmetric: bool = True
) -> tuple:
    """Quantize the input tensor to int8 and return the quantized tensor and scale, and maybe azp."""
    return _custom_ops.scaled_int8_quant(
        input,
        scale,
        azp,
        symmetric
    )
