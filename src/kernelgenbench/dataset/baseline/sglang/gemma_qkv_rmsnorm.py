"""
SGLang gemma_qkv_rmsnorm baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.gemma4_fused_ops import gemma_qkv_rmsnorm as _gemma_qkv_rmsnorm
except ModuleNotFoundError:
    _gemma_qkv_rmsnorm = None



def gemma_qkv_rmsnorm(q: torch.Tensor, k: typing.Optional[torch.Tensor], v: typing.Optional[torch.Tensor], q_weight: torch.Tensor, k_weight: typing.Optional[torch.Tensor], num_q_heads: int, num_kv_heads: int, head_dim: int, eps: float = 1e-6) -> None:
    """
    Args:
        q: torch.Tensor
        k: typing.Optional[torch.Tensor]
        v: typing.Optional[torch.Tensor]
        q_weight: torch.Tensor
        k_weight: typing.Optional[torch.Tensor]
        num_q_heads: int
        num_kv_heads: int
        head_dim: int
        eps: float
    Returns:
        None
    """
    _gemma_qkv_rmsnorm(q, k, v, q_weight, k_weight, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_dim=head_dim, eps=eps)

