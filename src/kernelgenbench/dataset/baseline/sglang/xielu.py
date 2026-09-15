"""
SGLang XIELU activation baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.activation import XIELU as _XIELU
except ModuleNotFoundError:
    _XIELU = None

_xielu_module = None

def _get_xielu_module():
    global _xielu_module
    if _xielu_module is None:
        _xielu_module = _XIELU().cuda()
    return _xielu_module

def xielu(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
    Returns:
        torch.Tensor
    """
    return _xielu_module(x)

