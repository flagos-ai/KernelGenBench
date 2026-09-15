"""
SGLang DeepseekScalingRotaryEmbedding baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.rotary_embedding.rope_variant import DeepseekScalingRotaryEmbedding as _DeepseekScalingRotaryEmbedding
except ModuleNotFoundError:
    _DeepseekScalingRotaryEmbedding = None

_deepseek_rope_module = None
_current_dsrope_config = None

def _get_deepseek_rope_module(head_size, rotary_dim, max_position_embeddings, base, scaling_factor):
    global _deepseek_rope_module, _current_dsrope_config
    key = (head_size, rotary_dim, max_position_embeddings, base, scaling_factor)
    if _current_dsrope_config != key:
        _deepseek_rope_module = _DeepseekScalingRotaryEmbedding(
            head_size=head_size, rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings, base=base,
            scaling_factor=scaling_factor
        ).cuda()
        _current_dsrope_config = key
    return _deepseek_rope_module

def deepseek_scaling_rotary_embedding(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor, offsets: typing.Optional[torch.Tensor] = None, head_size: int, rotary_dim: int, max_position_embeddings: int = 8192, base: float = 10000.0, scaling_factor: float = 1.0) -> typing.Tuple[torch.Tensor, torch.Tensor]:
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
    _deepseek_rope_module(q, k, positions, offsets)
    return q, k

