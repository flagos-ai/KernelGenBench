"""
SGLang gelu_and_mul_triton baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.elementwise import gelu_and_mul_triton as _gelu_and_mul_t
except ModuleNotFoundError:
    _gelu_and_mul_t = None



def gelu_and_mul_triton(hidden_states: torch.Tensor, scales: typing.Optional[torch.Tensor] = None, quantize: typing.Optional[str] = None, out: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Args:
        hidden_states: torch.Tensor
        scales: typing.Optional[torch.Tensor]
        quantize: typing.Optional[str]
        out: typing.Optional[torch.Tensor]
    Returns:
        torch.Tensor
    """
    return _gelu_and_mul_t(hidden_states, scales=scales, quantize=quantize, out=out)[0]

