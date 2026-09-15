"""
SGLang rms_norm_gated baseline (FLA fused norm+gate).
"""
import torch
import typing
try:
    from sglang.srt.layers.attention.fla.layernorm_gated import rms_norm_gated as _rms_norm_gated_fn
except ModuleNotFoundError:
    _rms_norm_gated_fn = None



def rms_norm_gated(x: torch.Tensor, weight: torch.Tensor, bias: typing.Optional[torch.Tensor] = None, z: typing.Optional[torch.Tensor] = None, eps: float = 1e-6, group_size: typing.Optional[int] = None, norm_before_gate: bool = True, activation: str = 'swish') -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
        weight: torch.Tensor
        bias: typing.Optional[torch.Tensor]
        z: typing.Optional[torch.Tensor]
        eps: float
        group_size: typing.Optional[int]
        norm_before_gate: bool
        activation: str
    Returns:
        torch.Tensor
    """
    return _rms_norm_gated_fn(x, weight, bias=bias, z=z, eps=eps, group_size=group_size, norm_before_gate=norm_before_gate, is_rms_norm=True, activation=activation)

