"""
vLLM marlin_int4_fp8_preprocess baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def marlin_int4_fp8_preprocess(
    qweight: torch.Tensor,
    qzeros_or_none: torch.Tensor | None = None,
    inplace: bool = False
) -> None:
    """Wrapper for vLLM marlin_int4_fp8_preprocess implementation."""
    return _custom_ops.marlin_int4_fp8_preprocess(
        qweight,
        qzeros_or_none,
        inplace
    )
