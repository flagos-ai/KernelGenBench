"""
vLLM ggml_moe_a8 baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def ggml_moe_a8(
    X: torch.Tensor,
    W: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int
) -> torch.Tensor:
    """Wrapper for vLLM ggml_moe_a8 implementation."""
    return _custom_ops.ggml_moe_a8(
        X, W, sorted_token_ids, expert_ids, num_tokens_post_padded,
        quant_type, row, top_k, tokens
    )
