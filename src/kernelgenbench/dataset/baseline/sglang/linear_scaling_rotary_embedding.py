"""
SGLang LinearScalingRotaryEmbedding baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.rotary_embedding.base import LinearScalingRotaryEmbedding as _LinearScalingRotaryEmbedding
except ModuleNotFoundError:
    _LinearScalingRotaryEmbedding = None

_linear_scaling_module = None
_current_ls_config = None

def _get_linear_scaling_module(head_size, rotary_dim, max_position_embeddings, base, scaling_factor):
    global _linear_scaling_module, _current_ls_config
    key = (head_size, rotary_dim, max_position_embeddings, base, scaling_factor)
    if _current_ls_config != key:
        _linear_scaling_module = _LinearScalingRotaryEmbedding(
            head_size=head_size, rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings, base=base,
            scaling_factor=scaling_factor
        ).cuda()
        _current_ls_config = key
    return _linear_scaling_module

def linear_scaling_rotary_embedding(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, offsets: typing.Optional[torch.Tensor] = None, head_size: int, rotary_dim: int, max_position_embeddings: int = 8192, base: float = 10000.0, scaling_factor: float = 1.0) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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
    _linear_scaling_module(q, k, positions, offsets)
    return q, k

