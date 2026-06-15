"""vLLM copy_blocks_mla baseline wrapper."""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def copy_blocks_mla(
    kv_caches: list,
    block_mapping: torch.Tensor
) -> None:
    _custom_ops.copy_blocks_mla(kv_caches, block_mapping)
