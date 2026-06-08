"""
vLLM gptq_shuffle baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def gptq_shuffle(
    q_weight: torch.Tensor,
    q_perm: torch.Tensor,
    bit: int
) -> None:
    """Wrapper for vLLM gptq_shuffle implementation."""
    _custom_ops.gptq_shuffle(
        q_weight,
        q_perm,
        bit
    )
