"""
vLLM cutlass_scaled_mm baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None
) -> torch.Tensor:
    """`cutlass_scaled_mm` implements a fused version of"""
    return _custom_ops.cutlass_scaled_mm(
        a,
        b,
        scale_a,
        scale_b,
        out_dtype,
        bias
    )
