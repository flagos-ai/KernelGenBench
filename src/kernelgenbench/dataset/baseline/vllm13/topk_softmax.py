"""
vLLM topk_softmax baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
) -> None:
    """Wrapper for vLLM topk_softmax implementation."""
    _custom_ops.topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
    )
