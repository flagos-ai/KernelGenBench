"""
vLLM fused_qk_norm_rope baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def fused_qk_norm_rope(
    qkv: torch.Tensor,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
    position_ids: torch.Tensor
) -> None:
    """Wrapper for vLLM fused_qk_norm_rope implementation."""
    _custom_ops.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        is_neox,
        position_ids
    )
