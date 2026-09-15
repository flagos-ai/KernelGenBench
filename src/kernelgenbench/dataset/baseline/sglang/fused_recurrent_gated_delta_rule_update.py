"""
SGLang fused_recurrent_gated_delta_rule_update baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule_update as _fused_rule_update
except ModuleNotFoundError:
    _fused_rule_update = None



def fused_recurrent_gated_delta_rule_update(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, scale: float, initial_state: torch.Tensor, cu_seqlens: typing.Optional[torch.Tensor] = None, initial_state_indices: typing.Optional[torch.Tensor] = None, intermediate_states: typing.Optional[torch.Tensor] = None, eagle_tree: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Args:
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        g: torch.Tensor
        beta: torch.Tensor
        scale: float
        initial_state: torch.Tensor
        cu_seqlens: typing.Optional[torch.Tensor]
        initial_state_indices: typing.Optional[torch.Tensor]
        intermediate_states: typing.Optional[torch.Tensor]
        eagle_tree: typing.Optional[torch.Tensor]
    Returns:
        torch.Tensor
    """
    return _fused_rule_update(q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=initial_state, cu_seqlens=cu_seqlens, initial_state_indices=initial_state_indices, intermediate_states=intermediate_states, eagle_tree=eagle_tree)

