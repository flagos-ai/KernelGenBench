"""
vLLM awq_gemm baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def awq_gemm(
    input: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    split_k_iters: int
) -> torch.Tensor:
    """Wrapper for vLLM awq_gemm implementation."""
    return _custom_ops.awq_gemm(
        input,
        qweight,
        scales,
        qzeros,
        split_k_iters
    )
