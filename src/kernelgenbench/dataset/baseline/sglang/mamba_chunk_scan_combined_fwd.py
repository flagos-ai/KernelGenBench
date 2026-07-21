"""
SGLang mamba_chunk_scan_combined_fwd baseline (Mamba-2 SSD).
"""
import torch
import typing
try:
    from sglang.srt.layers.attention.mamba.ops.ssd_combined import mamba_chunk_scan_combined_fwd as _mamba_scan
except ModuleNotFoundError:
    _mamba_scan = None



def mamba_chunk_scan_combined_fwd(x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, chunk_size: int = 64, D: typing.Optional[torch.Tensor] = None, z: typing.Optional[torch.Tensor] = None, dt_bias: typing.Optional[torch.Tensor] = None, initial_states: typing.Optional[torch.Tensor] = None, cu_seqlens: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
        dt: torch.Tensor
        A: torch.Tensor
        B: torch.Tensor
        C: torch.Tensor
        chunk_size: int
        D: typing.Optional[torch.Tensor]
        z: typing.Optional[torch.Tensor]
        dt_bias: typing.Optional[torch.Tensor]
        initial_states: typing.Optional[torch.Tensor]
        cu_seqlens: typing.Optional[torch.Tensor]
    Returns:
        torch.Tensor
    """
    return _mamba_scan(x, dt, A, B, C, chunk_size=chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, cu_seqlens=cu_seqlens)

