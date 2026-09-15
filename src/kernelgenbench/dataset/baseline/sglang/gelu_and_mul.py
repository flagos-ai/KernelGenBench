"""
SGLang GeluAndMul activation baseline.
"""
import torch
import typing
try:
    import sgl_kernel.ops.elementwise as _ops
except ModuleNotFoundError:
    _ops = None



def gelu_and_mul(x: torch.Tensor, approximate: str = 'tanh') -> torch.Tensor:
    """
    Args:
        x: torch.Tensor
        approximate: str
    Returns:
        torch.Tensor
    """
    x = x.contiguous()
    out = torch.empty(x.shape[0], x.shape[-1] // 2, dtype=x.dtype, device=x.device)
    _ops.gelu_and_mul(x, out, approximate)
    return out

