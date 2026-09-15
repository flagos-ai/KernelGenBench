"""
SGLang Gemma4RMSNorm baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.layernorm import Gemma4RMSNorm as _Gemma4RMSNorm
except ModuleNotFoundError:
    _Gemma4RMSNorm = None

_gemma4_rms_norm_module = None
_current_g4rn_config = None

def _get_gemma4_rms_norm_module(hidden_size, eps):
    global _gemma4_rms_norm_module, _current_g4rn_config
    key = (hidden_size, eps)
    if _current_g4rn_config != key:
        _gemma4_rms_norm_module = _Gemma4RMSNorm(hidden_size=hidden_size, eps=eps).cuda()
        _current_g4rn_config = key
    return _gemma4_rms_norm_module

def gemma4_rms_norm(x: torch.Tensor, hidden_size: int, eps: float = 1e-6, scale_shift: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
        hidden_size: int
        eps: float
        scale_shift: typing.Optional[torch.Tensor]
    Returns:
        torch.Tensor
    """
    return _gemma4_rms_norm_module(x, scale_shift)

