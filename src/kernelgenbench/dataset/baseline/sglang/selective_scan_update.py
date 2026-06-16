"""
SGLang selective_scan_update baseline (Mamba SSM).
"""
import torch
import typing
try:
    from sglang.srt.layers.attention.mamba.ops.mamba_ssm import selective_scan_update as _selective_scan_update
except ModuleNotFoundError:
    _selective_scan_update = None



def selective_scan_update(state: torch.Tensor, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: typing.Optional[torch.Tensor] = None, z: typing.Optional[torch.Tensor] = None, dt_bias: typing.Optional[torch.Tensor] = None, state_batch_indices: typing.Optional[torch.Tensor] = None) -> typing.Tuple[torch.Tensor, typling.Optional[torch.Tensor]]:
    """
    Args:
        state: torch.Tensor
        x: torch.Tensor
        dt: torch.Tensor
        A: torch.Tensor
        B: torch.Tensor
        C: torch.Tensor
        D: typing.Optional[torch.Tensor]
        z: typing.Optional[torch.Tensor]
        dt_bias: typing.Optional[torch.Tensor]
        state_batch_indices: typing.Optional[torch.Tensor]
    Returns:
        typing.Tuple[torch.Tensor, typling.Optional[torch.Tensor]]
    """
    return _selective_scan_update(state=state, x=x, dt=dt, A=A, B=B, C=C, D=D, z=z, dt_bias=dt_bias, state_batch_indices=state_batch_indices)

