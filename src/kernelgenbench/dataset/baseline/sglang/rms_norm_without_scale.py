"""
SGLang RMSNormWithoutScale baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.layernorm import RMSNormWithoutScale as _RMSNormWithoutScale
except ModuleNotFoundError:
    _RMSNormWithoutScale = None

_rms_norm_without_scale_module = None
_current_rnws_config = None

def _get_rms_norm_without_scale_module(hidden_size, eps):
    global _rms_norm_without_scale_module, _current_rnws_config
    key = (hidden_size, eps)
    if _current_rnws_config != key:
        _rms_norm_without_scale_module = _RMSNormWithoutScale(hidden_size=hidden_size, eps=eps).cuda()
        _current_rnws_config = key
    return _rms_norm_without_scale_module

def rms_norm_without_scale(x: torch.Tensor, hidden_size: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
        hidden_size: int
        eps: float
    Returns:
        torch.Tensor
    """
    return _rms_norm_without_scale_module(x)

