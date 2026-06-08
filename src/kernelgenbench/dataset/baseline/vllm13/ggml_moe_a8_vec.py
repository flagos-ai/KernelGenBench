"""
vLLM ggml_moe_a8_vec baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def ggml_moe_a8_vec(
    X: torch.Tensor,
    W: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int
) -> torch.Tensor:
    """Wrapper for vLLM ggml_moe_a8_vec implementation."""
    return _custom_ops.ggml_moe_a8_vec(
        X, W, topk_ids, top_k, quant_type, row, tokens
    )
