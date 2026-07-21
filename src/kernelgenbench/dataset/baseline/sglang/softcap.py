"""
SGLang Softcap baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.elementwise import Softcap as _Softcap
except ModuleNotFoundError:
    _Softcap = None

_softcap_module = None
_current_sc_config = None

def _get_softcap_module(softcap_val):
    global _softcap_module, _current_sc_config
    key = (softcap_val,)
    if _current_sc_config != key:
        _softcap_module = _Softcap(softcap_val).cuda()
        _current_sc_config = key
    return _softcap_module

def softcap(x: torch.Tensor, softcap: float = 50.0) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
        softcap: float
    Returns:
        torch.Tensor
    """
    return _softcap_module(x)

