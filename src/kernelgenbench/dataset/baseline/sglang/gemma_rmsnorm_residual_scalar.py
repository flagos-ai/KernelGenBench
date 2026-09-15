"""
SGLang gemma_rmsnorm_residual_scalar baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.gemma4_fused_ops import gemma_rmsnorm_residual_scalar as _gemma_residual_scalar
except ModuleNotFoundError:
    _gemma_residual_scalar = None



def gemma_rmsnorm_residual_scalar(x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor, scalar: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
        weight: torch.Tensor
        residual: torch.Tensor
        scalar: torch.Tensor
        eps: float
    Returns:
        torch.Tensor
    """
    return _gemma_residual_scalar(x, weight, residual, scalar, eps=eps)

