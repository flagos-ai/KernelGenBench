"""
SGLang DynamicNTKScalingRotaryEmbedding baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.rotary_embedding.rope_variant import DynamicNTKScalingRotaryEmbedding as _DynamicNTKScalingRotaryEmbedding
except ModuleNotFoundError:
    _DynamicNTKScalingRotaryEmbedding = None

_dynamic_ntk_module = None
_current_dntk_config = None

def _get_dynamic_ntk_module(head_size, rotary_dim, max_position_embeddings, base, scaling_factor):
    global _dynamic_ntk_module, _current_dntk_config
    key = (head_size, rotary_dim, max_position_embeddings, base, scaling_factor)
    if _current_dntk_config != key:
        _dynamic_ntk_module = _DynamicNTKScalingRotaryEmbedding(
            head_size=head_size, rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings, base=base,
            scaling_factor=scaling_factor
        ).cuda()
        _current_dntk_config = key
    return _dynamic_ntk_module

def dynamic_ntk_scaling_rotary_embedding(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, offsets: typing.Optional[torch.Tensor] = None, head_size: int, rotary_dim: int, max_position_embeddings: int = 8192, base: float = 10000.0, scaling_factor: float = 1.0) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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
        scaling_factor: float
    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]
    """
    q = query.clone()
    k = key.clone()
    _dynamic_ntk_module(q, k, positions, offsets)
    return q, k

