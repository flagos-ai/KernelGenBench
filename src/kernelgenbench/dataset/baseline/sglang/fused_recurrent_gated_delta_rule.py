"""
SGLang fused_recurrent_gated_delta_rule baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.attention.fla.fused_recurrent import fused_recurrent_gated_delta_rule as _fused_rule
except ModuleNotFoundError:
    _fused_rule = None



def fused_recurrent_gated_delta_rule(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, scale: float, initial_state: typing.Optional[torch.Tensor] = None, output_final_state: bool = False, cu_seqlens: typing.Optional[torch.Tensor] = None) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]:
    """
    Args:
        q: torch.Tensor
        k: torch.Tensor
        v: torch.Tensor
        g: torch.Tensor
        beta: torch.Tensor
        scale: float
        initial_state: typing.Optional[torch.Tensor]
        output_final_state: bool
        cu_seqlens: typing.Optional[torch.Tensor]
    Returns:
        typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]
    """
    return _fused_rule(q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=initial_state, output_final_state=output_final_state, cu_seqlens=cu_seqlens)

