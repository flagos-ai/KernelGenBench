"""
SGLang triton_mrope_fused baseline (Triton fused multimodal RoPE).
"""
import torch
import typing
try:
    from sglang.srt.layers.rotary_embedding.triton_kernels import triton_mrope_fused as _triton_mrope_fused
except ModuleNotFoundError:
    _triton_mrope_fused = None



def triton_mrope_fused(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, cos_sin_cache: torch.Tensor, head_size: int, rotary_dim: int, mrope_section: typing.Optional[typing.List[int]] = None) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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
    _triton_mrope_fused(q, k, cos_sin_cache, positions, mrope_section=[head_size]*3, head_size=head_size, rotary_dim=rotary_dim, mrope_interleaved=False, mrope_interleaved_glm=False, is_neox_style=True, axis_map=None)
    return q, k

