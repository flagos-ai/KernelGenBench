"""
SGLang FusedDualResidualRMSNorm baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.elementwise import FusedDualResidualRMSNorm as _FusedDualResidualRMSNorm
except ModuleNotFoundError:
    _FusedDualResidualRMSNorm = None

_fused_dual_residual_rmsnorm_module = None
_current_fdrn_config = None

def _get_fused_dual_residual_rmsnorm_module(hidden_size_1, hidden_size_2, eps):
    global _fused_dual_residual_rmsnorm_module, _current_fdrn_config
    key = (hidden_size_1, hidden_size_2, eps)
    if _current_fdrn_config != key:
        _fused_dual_residual_rmsnorm_module = _FusedDualResidualRMSNorm(hidden_size_1=hidden_size_1, hidden_size_2=hidden_size_2, eps=eps).cuda()
        _current_fdrn_config = key
    return _fused_dual_residual_rmsnorm_module

def fused_dual_residual_rmsnorm(x: torch.Tensor, residual: torch.Tensor, hidden_size_1: int, hidden_size_2: int, eps: float = 1e-6) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x: torch.Tensor
        residual: torch.Tensor
        hidden_size_1: int
        hidden_size_2: int
        eps: float
    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]
    """
    return _fused_dual_residual_rmsnorm_module(x, residual)

