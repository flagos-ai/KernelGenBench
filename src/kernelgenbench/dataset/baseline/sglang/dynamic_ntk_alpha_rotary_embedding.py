"""
SGLang DynamicNTKAlphaRotaryEmbedding baseline.
Source: sglang.srt.layers.rotary_embedding.rope_variant.DynamicNTKAlphaRotaryEmbedding(head_size,rotary_dim,max_positions,base,is_neox,dtype,scaling_alpha)
"""
import torch
import typing

try:
    from sglang.srt.layers.rotary_embedding.rope_variant import DynamicNTKAlphaRotaryEmbedding as _DynamicNTKAlphaRotaryEmbedding
except ModuleNotFoundError:
    _DynamicNTKAlphaRotaryEmbedding = None

_module = None
_current_config = None


def _get_module(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, scaling_alpha):
    global _module, _current_config
    key = (head_size, rotary_dim, max_position_embeddings, base, is_neox_style, str(dtype), scaling_alpha)
    if _current_config != key:
        _module = _DynamicNTKAlphaRotaryEmbedding(
            head_size=head_size, rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings, base=base,
            is_neox_style=is_neox_style, dtype=dtype or torch.bfloat16,
            scaling_alpha=scaling_alpha
        ).cuda()
        _current_config = key
    return _module


def dynamic_ntk_alpha_rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    offsets: typing.Optional[torch.Tensor] = None,
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int = 8192,
    base: float = 10000.0,
    is_neox_style: bool = True,
    scaling_alpha: float = 1.0,
    dtype: torch.dtype = None,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic NTK alpha-scaled RoPE. In-place modifies q, k."""
    module = _get_module(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, scaling_alpha)
    q = query.clone()
    k = key.clone()
    module.forward_cuda(positions, q, k, offsets)
    return q, k
