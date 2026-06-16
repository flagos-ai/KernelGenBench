"""
SGLang experts_combine_triton baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.elementwise import experts_combine_triton as _experts_combine
except ModuleNotFoundError:
    _experts_combine = None



def experts_combine_triton(moe_hidden_states: torch.Tensor, mlp_hidden_states: torch.Tensor, output_buffer: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Args:
        moe_hidden_states: torch.Tensor
        mlp_hidden_states: torch.Tensor
        output_buffer: typing.Optional[torch.Tensor]
    Returns:
        torch.Tensor
    """
    return _experts_combine(moe_hidden_states, mlp_hidden_states, output_buffer=output_buffer)

