import torch
try:
    from vllm import _custom_ops as ops
except ModuleNotFoundError:
    ops = None


def apply_repetition_penalties_cuda(
    logits: torch.Tensor,
    prompt_mask: torch.Tensor,
    output_mask: torch.Tensor,
    repetition_penalties: torch.Tensor,
) -> None:
    ops.apply_repetition_penalties_cuda(logits, prompt_mask, output_mask, repetition_penalties)
