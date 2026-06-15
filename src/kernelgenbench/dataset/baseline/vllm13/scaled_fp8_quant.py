"""
vLLM scaled_fp8_quant baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def scaled_fp8_quant(
    input: torch.Tensor,
    scale: torch.Tensor | None = None,
    num_token_padding: int | None = None,
    scale_ub: torch.Tensor | None = None,
    use_per_token_if_dynamic: bool = False,
    output: torch.Tensor | None = None,
) -> tuple:
    """Quantize input tensor to FP8 and return quantized tensor and scale."""
    return _custom_ops.scaled_fp8_quant(
        input,
        scale,
        num_token_padding,
        scale_ub,
        use_per_token_if_dynamic,
        output,
    )
