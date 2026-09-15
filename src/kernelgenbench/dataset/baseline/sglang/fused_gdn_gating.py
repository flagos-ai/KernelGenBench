"""
SGLang fused_gdn_gating baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating as _fused_gdn
except ModuleNotFoundError:
    _fused_gdn = None



def fused_gdn_gating(A_log: torch.Tensor, a: torch.Tensor, b: torch.Tensor, dt_bias: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        A_log: torch.Tensor
        a: torch.Tensor
        b: torch.Tensor
        dt_bias: torch.Tensor
        beta: float
        threshold: float
    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]
    """
    return _fused_gdn(A_log=A_log, a=a, b=b, dt_bias=dt_bias, beta=beta, threshold=threshold)

