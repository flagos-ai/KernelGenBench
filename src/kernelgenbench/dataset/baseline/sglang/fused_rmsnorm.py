"""
SGLang fused_rmsnorm baseline (pure Triton).
"""
import torch
import typing
try:
    from sglang.srt.layers.elementwise import fused_rmsnorm as _fused_rmsnorm_fn
except ModuleNotFoundError:
    _fused_rmsnorm_fn = None



def fused_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, autotune: bool = False, inplace: bool = False) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
        weight: torch.Tensor
        eps: float
        autotune: bool
        inplace: bool
    Returns:
        torch.Tensor
    """
    return _fused_rmsnorm_fn(x, weight, eps, autotune=autotune, inplace=inplace)

