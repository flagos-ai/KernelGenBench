"""
SGLang TopK MoE router baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.moe.topk import TopK as _TopK
except ModuleNotFoundError:
    _TopK = None

_topk_module = None
_current_topk_config = None

def _get_topk_module(topk, renormalize):
    global _topk_module, _current_topk_config
    key = (topk, renormalize)
    if _current_topk_config != key:
        _topk_module = _TopK(topk=topk, renormalize=renormalize).cuda()
        _current_topk_config = key
    return _topk_module

def topk(hidden_states: torch.Tensor, router_logits: torch.Tensor, topk: int = 8, renormalize: bool = True) -> "typing.Any":
    """
    Args:
        hidden_states: torch.Tensor
        router_logits: torch.Tensor
        topk: int
        renormalize: bool
    Returns:
        "typing.Any"
    """
    return _topk_module(hidden_states, router_logits)

