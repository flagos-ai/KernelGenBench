"""
SGLang RotaryEmbedding baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.rotary_embedding import RotaryEmbedding as _RotaryEmbedding
except ModuleNotFoundError:
    _RotaryEmbedding = None

_rotary_embedding_module = None
_current_rope_config = None

def _get_rotary_module(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype):
    global _rotary_embedding_module, _current_rope_config
    key = (head_size, rotary_dim, max_position_embeddings, base, is_neox_style, str(dtype))
    if _current_rope_config != key:
        _rotary_embedding_module = _RotaryEmbedding(
            head_size=head_size, rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings, base=base,
            is_neox_style=is_neox_style, dtype=dtype or torch.bfloat16
        ).cuda()
        _current_rope_config = key
    return _rotary_embedding_module

def rotary_embedding(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, offsets: typing.Optional[torch.Tensor] = None, head_size: int, rotary_dim: int, max_position_embeddings: int = 8192, base: float = 10000.0, is_neox_style: bool = True, dtype: torch.dtype = None) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        positions: torch.Tensor
        query: torch.Tensor
        key: torch.Tensor
        offsets: typing.Optional[torch.Tensor]
        head_size: int
        rotary_dim: int
        max_position_embeddings: int
        base: float
        is_neox_style: bool
        dtype: torch.dtype
    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]
    """
    q = query.clone()
    k = key.clone()
    _rotary_embedding_module(q, k, positions, offsets)
    return q, k

