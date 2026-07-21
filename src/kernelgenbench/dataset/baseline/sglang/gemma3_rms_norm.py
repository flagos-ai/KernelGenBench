"""
SGLang Gemma3RMSNorm baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.layernorm import Gemma3RMSNorm as _Gemma3RMSNorm
except ModuleNotFoundError:
    _Gemma3RMSNorm = None

_gemma3_rms_norm_module = None
_current_g3rn_config = None

def _get_gemma3_rms_norm_module(hidden_size, eps):
    global _gemma3_rms_norm_module, _current_g3rn_config
    key = (hidden_size, eps)
    if _current_g3rn_config != key:
        _gemma3_rms_norm_module = _Gemma3RMSNorm(hidden_size=hidden_size, eps=eps).cuda()
        _current_g3rn_config = key
    return _gemma3_rms_norm_module

def gemma3_rms_norm(x: torch.Tensor, hidden_size: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
        hidden_size: int
        eps: float
    Returns:
        torch.Tensor
    """
    return _gemma3_rms_norm_module(x)

