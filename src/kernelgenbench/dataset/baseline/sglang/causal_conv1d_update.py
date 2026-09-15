"""
SGLang causal_conv1d_update baseline.
"""
import torch
import typing
try:
    import sgl_kernel as _sgl_kernel
    _causal_conv1d_update = _sgl_kernel.causal_conv1d_update
except (ModuleNotFoundError, AttributeError):
    _causal_conv1d_update = None



def causal_conv1d_update(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor, bias: typing.Optional[torch.Tensor] = None, activation: str = 'silu') -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: torch.Tensor
        conv_state: torch.Tensor
        weight: torch.Tensor
        bias: typing.Optional[torch.Tensor]
        activation: str
    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]
    """
    return _causal_conv1d_update(x, conv_state, weight, bias=bias, activation=activation)

