"""
vLLM ggml_dequantize baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def ggml_dequantize(
    W: torch.Tensor,
    quant_type: int,
    m: int,
    n: int,
    dtype: torch.dtype
) -> torch.Tensor:
    """Wrapper for vLLM ggml_dequantize implementation."""
    return _custom_ops.ggml_dequantize(
        W, quant_type, m, n, dtype
    )
