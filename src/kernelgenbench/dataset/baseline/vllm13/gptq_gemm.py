"""
vLLM gptq_gemm baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def gptq_gemm(
    a: torch.Tensor,
    b_q_weight: torch.Tensor,
    b_gptq_qzeros: torch.Tensor,
    b_gptq_scales: torch.Tensor,
    b_g_idx: torch.Tensor,
    use_exllama: bool,
    use_v2_format: bool,
    bit: int
) -> torch.Tensor:
    """Wrapper for vLLM gptq_gemm implementation."""
    return _custom_ops.gptq_gemm(
        a,
        b_q_weight,
        b_gptq_qzeros,
        b_gptq_scales,
        b_g_idx,
        use_exllama,
        use_v2_format,
        bit
    )
