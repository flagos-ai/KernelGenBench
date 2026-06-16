"""
SGLang DualChunkRotaryEmbedding baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.rotary_embedding.rope_variant import DualChunkRotaryEmbedding as _DualChunkRotaryEmbedding
except ModuleNotFoundError:
    _DualChunkRotaryEmbedding = None

_dual_chunk_module = None
_current_dc_config = None

def _get_dual_chunk_module(head_size, rotary_dim, max_position_embeddings, base):
    global _dual_chunk_module, _current_dc_config
    key = (head_size, rotary_dim, max_position_embeddings, base)
    if _current_dc_config != key:
        _dual_chunk_module = _DualChunkRotaryEmbedding(
            head_size=head_size, rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings, base=base
        ).cuda()
        _current_dc_config = key
    return _dual_chunk_module

def dual_chunk_rotary_embedding(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, offsets: typing.Optional[torch.Tensor] = None, head_size: int, rotary_dim: int, max_position_embeddings: int = 8192, base: float = 10000.0) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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
    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]
    """
    q = query.clone()
    k = key.clone()
    _dual_chunk_module(q, k, positions, offsets)
    return q, k

