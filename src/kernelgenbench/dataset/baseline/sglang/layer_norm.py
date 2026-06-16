"""
SGLang LayerNorm baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.layernorm import LayerNorm as _LayerNorm
except ModuleNotFoundError:
    _LayerNorm = None

_layer_norm_module = None
_current_ln_config = None

def _get_layer_norm_module(normalized_shape, eps, elementwise_affine):
    global _layer_norm_module, _current_ln_config
    key = (normalized_shape, eps, elementwise_affine)
    if _current_ln_config != key:
        _layer_norm_module = _LayerNorm(normalized_shape=normalized_shape, eps=eps, elementwise_affine=elementwise_affine).cuda()
        _current_ln_config = key
    return _layer_norm_module

def layer_norm(x: torch.Tensor, normalized_shape: int, eps: float = 1e-5, elementwise_affine: bool = True) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
        normalized_shape: int
        eps: float
        elementwise_affine: bool
    Returns:
        torch.Tensor
    """
    return _layer_norm_module(x)

