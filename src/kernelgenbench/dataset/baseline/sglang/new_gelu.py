"""
SGLang NewGELU activation baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.activation import NewGELU as _NewGELU
except ModuleNotFoundError:
    _NewGELU = None

_new_gelu_module = None

def _get_new_gelu_module():
    global _new_gelu_module
    if _new_gelu_module is None:
        _new_gelu_module = _NewGELU().cuda()
    return _new_gelu_module

def new_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
    Returns:
        torch.Tensor
    """
    return _new_gelu_module(x)

