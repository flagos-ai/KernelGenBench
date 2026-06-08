"""
vLLM hadacore_transform baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def hadacore_transform(
    x: torch.Tensor,
    inplace: bool = True
) -> torch.Tensor:
    """Perform Hadamard transforms using [Hadacore](https://arxiv.org/abs/2412.08832)"""
    return _custom_ops.hadacore_transform(
        x,
        inplace
    )
