"""
SGLang Llama3RotaryEmbedding baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.rotary_embedding.rope_variant import Llama3RotaryEmbedding as _Llama3RotaryEmbedding
except ModuleNotFoundError:
    _Llama3RotaryEmbedding = None

_llama3_rope_module = None
_current_l3rope_config = None

def _get_llama3_rope_module(head_size, rotary_dim, max_position_embeddings, base, factor, low_freq_factor, high_freq_factor, original_max_position_embeddings):
    global _llama3_rope_module, _current_l3rope_config
    key = (head_size, rotary_dim, max_position_embeddings, base, factor, low_freq_factor, high_freq_factor, original_max_position_embeddings)
    if _current_l3rope_config != key:
        _llama3_rope_module = _Llama3RotaryEmbedding(
            head_size=head_size, rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings, base=base,
            factor=factor, low_freq_factor=low_freq_factor,
            high_freq_factor=high_freq_factor,
            original_max_position_embeddings=original_max_position_embeddings
        ).cuda()
        _current_l3rope_config = key
    return _llama3_rope_module

def llama3_rotary_embedding(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, offsets: typing.Optional[torch.Tensor] = None, head_size: int, rotary_dim: int, max_position_embeddings: int = 8192, base: float = 500000.0, factor: float = 8.0, low_freq_factor: float = 1.0, high_freq_factor: float = 4.0, original_max_position_embeddings: int = 8192) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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
        factor: float
        low_freq_factor: float
        high_freq_factor: float
        original_max_position_embeddings: int
    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]
    """
    q = query.clone()
    k = key.clone()
    _llama3_rope_module(q, k, positions, offsets)
    return q, k

