"""
SGLang fused_sigmoid_gating_delta_rule_update baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update as _fused_sig_rule_update
except ModuleNotFoundError:
    _fused_sig_rule_update = None



def fused_sigmoid_gating_delta_rule_update(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, b: torch.Tensor, scale: float, initial_state: torch.Tensor, cu_seqlens: typing.Optional[torch.Tensor] = None, initial_state_indices: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Args:
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        A_log: torch.Tensor
        a: torch.Tensor
        dt_bias: torch.Tensor
        b: torch.Tensor
        scale: float
        initial_state: torch.Tensor
        cu_seqlens: typing.Optional[torch.Tensor]
        initial_state_indices: typing.Optional[torch.Tensor]
    Returns:
        torch.Tensor
    """
    return _fused_sig_rule_update(q=q, k=k, v=v, A_log=A_log, a=a, dt_bias=dt_bias, b=b, scale=scale, initial_state=initial_state, cu_seqlens=cu_seqlens, initial_state_indices=initial_state_indices)

