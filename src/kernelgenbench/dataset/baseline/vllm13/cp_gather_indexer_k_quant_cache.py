"""
vLLM cp_gather_indexer_k_quant_cache baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def cp_gather_indexer_k_quant_cache(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    dst_scale: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor
) -> None:
    """Wrapper for vLLM cp_gather_indexer_k_quant_cache implementation."""
    _custom_ops.cp_gather_indexer_k_quant_cache(
        kv_cache,
        dst_k,
        dst_scale,
        block_table,
        cu_seq_lens
    )
