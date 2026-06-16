"""
SGLang Phi3LongRoPEScaledRotaryEmbedding baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.rotary_embedding.rope_variant import Phi3LongRoPEScaledRotaryEmbedding as _Phi3LongRoPEScaledRotaryEmbedding
except ModuleNotFoundError:
    _Phi3LongRoPEScaledRotaryEmbedding = None

_phi3_rope_module = None
_current_phi3_config = None

def _get_phi3_rope_module(head_size, rotary_dim, max_position_embeddings, base, short_factor, long_factor, original_max_position_embeddings):
    global _phi3_rope_module, _current_phi3_config
    key = (head_size, rotary_dim, max_position_embeddings, base, tuple(short_factor or []), tuple(long_factor or []), original_max_position_embeddings)
    if _current_phi3_config != key:
        _phi3_rope_module = _Phi3LongRoPEScaledRotaryEmbedding(
            head_size=head_size, rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings, base=base,
            short_factor=short_factor, long_factor=long_factor,
            original_max_position_embeddings=original_max_position_embeddings
        ).cuda()
        _current_phi3_config = key
    return _phi3_rope_module

def phi3_long_rope_scaled_rotary_embedding(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, offsets: typing.Optional[torch.Tensor] = None, head_size: int, rotary_dim: int, max_position_embeddings: int = 131072, base: float = 10000.0, short_factor: typing.List[float] = None, long_factor: typing.List[float] = None, original_max_position_embeddings: int = 4096) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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
        short_factor: typing.List[float]
        long_factor: typing.List[float]
        original_max_position_embeddings: int
    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]
    """
    q = query.clone()
    k = key.clone()
    _phi3_rope_module(q, k, positions, offsets)
    return q, k

