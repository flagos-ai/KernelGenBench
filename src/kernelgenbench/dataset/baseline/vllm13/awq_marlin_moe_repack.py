"""vLLM awq_marlin_moe_repack baseline wrapper."""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def awq_marlin_moe_repack(
    b_q_weight: torch.Tensor,
    perm: torch.Tensor,
    size_k: int,
    size_n: int,
    num_bits: int,
    is_a_8bit: bool = False
) -> torch.Tensor:
    return _custom_ops.awq_marlin_moe_repack(
        b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit
    )
