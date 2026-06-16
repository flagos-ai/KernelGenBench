"""
SGLang triton_ernie45_rope_fused_inplace baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.rotary_embedding.triton_kernels import triton_ernie45_rope_fused_inplace as _triton_ernie45_rope_fused
except ModuleNotFoundError:
    _triton_ernie45_rope_fused = None



def triton_ernie45_rope_fused(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, cos_sin_cache: torch.Tensor, head_size: int, rotary_dim: int, mrope_section: typing.Optional[typing.List[int]] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        positions: torch.Tensor
        query: torch.Tensor
        key: torch.Tensor
        cos_sin_cache: torch.Tensor
        head_size: int
        rotary_dim: int
        mrope_section: typing.Optional[typing.List[int]]
    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]
    """
    q = query.clone()
    k = key.clone()
    _triton_ernie45_rope_fused(q, k, cos_sin_cache, positions, mrope_section=[head_size]*3, head_size=head_size, rotary_dim=rotary_dim, is_neox_style=True)
    return q, k

