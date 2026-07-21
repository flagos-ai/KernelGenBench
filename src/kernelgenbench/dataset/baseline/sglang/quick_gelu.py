"""
SGLang QuickGELU activation baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.activation import QuickGELU as _QuickGELU
except ModuleNotFoundError:
    _QuickGELU = None

_quick_gelu_module = None

def _get_quick_gelu_module():
    global _quick_gelu_module
    if _quick_gelu_module is None:
        _quick_gelu_module = _QuickGELU().cuda()
    return _quick_gelu_module

def quick_gelu(x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
    Returns:
        torch.Tensor
    """
    return _quick_gelu_module(x)

