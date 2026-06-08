"""
vLLM gather_and_maybe_dequant_cache baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def gather_and_maybe_dequant_cache(
    src_cache: torch.Tensor,
    dst: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    token_to_seq: torch.Tensor,
    num_tokens: int,
    kv_cache_dtype: str,
    scale: torch.Tensor,
    seq_starts: torch.Tensor | None = None
) -> None:
    """Wrapper for vLLM gather_and_maybe_dequant_cache implementation."""
    _custom_ops.gather_and_maybe_dequant_cache(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        kv_cache_dtype,
        scale,
        seq_starts
    )
