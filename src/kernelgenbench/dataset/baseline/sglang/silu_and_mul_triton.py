"""
SGLang silu_and_mul_triton baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.elementwise import silu_and_mul_triton as _silu_and_mul_t
except ModuleNotFoundError:
    _silu_and_mul_t = None



def silu_and_mul_triton(hidden_states: torch.Tensor, scales: typing.Optional[torch.Tensor] = None, quantize: typing.Optional[str] = None, out: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Args:
        hidden_states: torch.Tensor
        scales: typing.Optional[torch.Tensor]
        quantize: typing.Optional[str]
        out: typing.Optional[torch.Tensor]
    Returns:
        torch.Tensor
    """
    return _silu_and_mul_t(hidden_states, scales=scales, quantize=quantize, out=out)[0]

