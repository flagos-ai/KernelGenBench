"""
vLLM ggml_mul_mat_a8 baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def ggml_mul_mat_a8(
    W: torch.Tensor,
    X: torch.Tensor,
    quant_type: int,
    row: int
) -> torch.Tensor:
    """Wrapper for vLLM ggml_mul_mat_a8 implementation."""
    return _custom_ops.ggml_mul_mat_a8(
        W,
        X,
        quant_type,
        row
    )
