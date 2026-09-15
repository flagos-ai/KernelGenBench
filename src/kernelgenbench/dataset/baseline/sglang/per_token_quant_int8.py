"""
SGLang per_token_quant_int8 baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8 as _per_token_quant
except ModuleNotFoundError:
    _per_token_quant = None



def per_token_quant_int8(x: torch.Tensor, scale_dtype: torch.dtype = torch.float32) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: torch.Tensor
        scale_dtype: torch.dtype
    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]
    """
    return _per_token_quant(x, scale_dtype=scale_dtype)

