"""
SGLang l2norm baseline (FLA Triton kernel).
"""
import torch
import typing
try:
    from sglang.srt.layers.attention.fla.l2norm import l2norm as _l2norm_fn
except ModuleNotFoundError:
    _l2norm_fn = None



def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
        eps: float
    Returns:
        torch.Tensor
    """
    return _l2norm_fn(x, eps=eps)

