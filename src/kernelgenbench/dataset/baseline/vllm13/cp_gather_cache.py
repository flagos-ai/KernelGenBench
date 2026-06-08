"""
vLLM cp_gather_cache baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def cp_gather_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    batch_size: int,
    seq_starts: torch.Tensor | None = None
) -> None:
    """Wrapper for vLLM cp_gather_cache implementation."""
    _custom_ops.cp_gather_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        batch_size,
        seq_starts
    )
