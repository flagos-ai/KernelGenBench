"""
SGLang MRotaryEmbedding baseline (multimodal RoPE).
"""
import torch
import typing
try:
    from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding as _MRotaryEmbedding
except ModuleNotFoundError:
    _MRotaryEmbedding = None

_mrotary_module = None
_current_mrope_config = None

def _get_mrotary_module(head_size, rotary_dim, max_position_embeddings, base, is_neox_style, mrope_section, mrope_interleaved, dtype):
    global _mrotary_module, _current_mrope_config
    key = (head_size, rotary_dim, max_position_embeddings, base, is_neox_style, tuple(mrope_section or []), mrope_interleaved, str(dtype))
    if _current_mrope_config != key:
        _mrotary_module = _MRotaryEmbedding(
            head_size=head_size, rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings, base=base,
            is_neox_style=is_neox_style, mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved, dtype=dtype or torch.bfloat16
        ).cuda()
        _current_mrope_config = key
    return _mrotary_module

def mrotary_embedding(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, head_size: int, rotary_dim: int, max_position_embeddings: int = 8192, base: float = 10000.0, is_neox_style: bool = True, mrope_section: typing.Optional[typing.List[int]] = None, mrope_interleaved: bool = False, dtype: torch.dtype = None) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        positions: torch.Tensor
        query: torch.Tensor
        key: torch.Tensor
        head_size: int
        rotary_dim: int
        max_position_embeddings: int
        base: float
        is_neox_style: bool
        mrope_section: typing.Optional[typing.List[int]]
        mrope_interleaved: bool
        dtype: torch.dtype
    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]
    """
    q = query.clone()
    k = key.clone()
    _mrotary_module.forward_cuda(positions, q, k)
    return q, k

