"""
SGLang layer_norm_gated_fwd baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.attention.fla.fused_norm_gate import layer_norm_gated_fwd as _layer_norm_gated
except ModuleNotFoundError:
    _layer_norm_gated = None



def layer_norm_gated_fwd(x: torch.Tensor, g: torch.Tensor, weight: torch.Tensor, bias: typing.Optional[torch.Tensor] = None, activation: str = 'swish', eps: float = 1e-5, residual: typing.Optional[torch.Tensor] = None) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, typling.Optional[torch.Tensor]]:
    """
    Args:
        x: torch.Tensor
        g: torch.Tensor
        weight: torch.Tensor
        bias: typing.Optional[torch.Tensor]
        activation: str
        eps: float
        residual: typing.Optional[torch.Tensor]
    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, typling.Optional[torch.Tensor]]
    """
    return _layer_norm_gated(x, g, weight, bias, activation=activation, eps=eps, residual=residual)

