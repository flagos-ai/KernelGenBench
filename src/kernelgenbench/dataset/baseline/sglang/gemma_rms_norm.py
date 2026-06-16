"""
SGLang GemmaRMSNorm baseline.
"""
import torch
import typing
try:
    import sgl_kernel.ops.norm as _ops
except ModuleNotFoundError:
    _ops = None



def gemma_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6, residual: typing.Optional[torch.Tensor] = None, out_residual: typing.Optional[torch.Tensor] = None) -> typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]:
    """
    Args:
        x: torch.Tensor
        weight: torch.Tensor
        eps: float
        residual: typing.Optional[torch.Tensor]
        out_residual: typing.Optional[torch.Tensor]
    Returns:
        typing.Union[torch.Tensor, typing.Tuple[torch.Tensor, torch.Tensor]]
    """
    x = x.contiguous()
    out = torch.empty_like(x)
    residual_out = None
    if residual is not None:
        residual_c = residual.contiguous()
        if out_residual is None:
            out_residual = torch.empty_like(residual_c)
        _ops.gemma_fused_add_rmsnorm(x, residual_c, weight, eps, out, out_residual)
        residual_out = out_residual
        return out, residual_out
    _ops.gemma_rmsnorm(out, x, weight, eps)
    if residual is None:
        return out
    return (out,)

