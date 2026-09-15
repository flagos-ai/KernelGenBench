"""
SGLang Mixer2RMSNormGated baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.attention.mamba.mixer2_rms_norm_gated import Mixer2RMSNormGated as _Mixer2RMSNormGated
except ModuleNotFoundError:
    _Mixer2RMSNormGated = None

_mixer2_module = None
_current_mixer2_config = None

def _get_mixer2_module(hidden_size, eps):
    global _mixer2_module, _current_mixer2_config
    key = (hidden_size, eps)
    if _current_mixer2_config != key:
        _mixer2_module = _Mixer2RMSNormGated(hidden_size=hidden_size, eps=eps).cuda()
        _current_mixer2_config = key
    return _mixer2_module

def mixer2_rms_norm_gated(x: torch.Tensor, gate: torch.Tensor, hidden_size: int, eps: float = 1e-6) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
        gate: torch.Tensor
        hidden_size: int
        eps: float
    Returns:
        torch.Tensor
    """
    return _mixer2_module(x, gate)

