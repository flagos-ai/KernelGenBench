"""
SGLang causal_conv1d_fn baseline.
"""
import torch
import typing
try:
    import sgl_kernel as _sgl_kernel
    _causal_conv1d_fn = _sgl_kernel.causal_conv1d_fwd
except (ModuleNotFoundError, AttributeError):
    _causal_conv1d_fn = None



def causal_conv1d_fn(x: torch.Tensor, weight: torch.Tensor, bias: typing.Optional[torch.Tensor] = None, query_start_loc: typing.Optional[torch.Tensor] = None, cache_indices: typing.Optional[torch.Tensor] = None, has_initial_state: typing.Optional[torch.Tensor] = None, conv_states: typing.Optional[torch.Tensor] = None, activation: str = 'silu') -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]:
    """
    Args:
        x: torch.Tensor
        weight: torch.Tensor
        bias: typing.Optional[torch.Tensor]
        query_start_loc: typing.Optional[torch.Tensor]
        cache_indices: typing.Optional[torch.Tensor]
        has_initial_state: typing.Optional[torch.Tensor]
        conv_states: typing.Optional[torch.Tensor]
        activation: str
    Returns:
        typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]
    """
    return _causal_conv1d_fn(x, weight, bias=bias, query_start_loc=query_start_loc, cache_indices=cache_indices, has_initial_state=has_initial_state, conv_states=conv_states, activation=activation)

