"""
vLLM allspark_w8a16_gemm baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def allspark_w8a16_gemm(
    a: torch.Tensor,
    b_qweight: torch.Tensor,
    b_scales: torch.Tensor,
    b_qzeros: torch.Tensor | None,
    n: int,
    group_size: int,
    sm_count: int,
    sm_version: int,
    CUBLAS_M_THRESHOLD: int,
    has_zp: bool,
    n32k16_reorder: bool
) -> torch.Tensor:
    """Wrapper for vLLM allspark_w8a16_gemm implementation."""
    return _custom_ops.allspark_w8a16_gemm(
        a,
        b_qweight,
        b_scales,
        b_qzeros,
        n,
        group_size,
        sm_count,
        sm_version,
        CUBLAS_M_THRESHOLD,
        has_zp,
        n32k16_reorder
    )
