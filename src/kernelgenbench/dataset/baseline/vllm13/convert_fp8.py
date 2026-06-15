"""
vLLM convert_fp8 baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def convert_fp8(
    output: torch.Tensor,
    input: torch.Tensor,
    scale: float = 1.0,
    kv_dtype: str = 'fp8'
) -> None:
    """Wrapper for vLLM convert_fp8 implementation."""
    _custom_ops.convert_fp8(
        output,
        input,
        scale,
        kv_dtype
    )
