"""
vLLM gptq_marlin_24_gemm baseline wrapper.
"""
import torch
try:
    import vllm
except ModuleNotFoundError:
    vllm = None
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def gptq_marlin_24_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_meta: torch.Tensor,
    b_scales: torch.Tensor,
    workspace: torch.Tensor,
    b_q_type: vllm.scalar_type.ScalarType,
    size_m: int,
    size_n: int,
    size_k: int
) -> torch.Tensor:
    """Wrapper for vLLM gptq_marlin_24_gemm implementation."""
    return _custom_ops.gptq_marlin_24_gemm(
        a,
        b_q_weight,
        b_meta,
        b_scales,
        workspace,
        b_q_type,
        size_m,
        size_n,
        size_k
    )
