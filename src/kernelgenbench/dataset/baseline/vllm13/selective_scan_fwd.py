"""
vLLM selective_scan_fwd baseline wrapper.
"""
import torch
try:
    from vllm import _custom_ops
except ModuleNotFoundError:
    _custom_ops = None


def selective_scan_fwd(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D_: torch.Tensor | None,
    z_: torch.Tensor | None,
    delta_bias_: torch.Tensor | None,
    delta_softplus: bool,
    query_start_loc: torch.Tensor | None,
    cache_indices: torch.Tensor | None,
    has_initial_state: torch.Tensor | None,
    ssm_states: torch.Tensor,
    pad_slot_id: int,
    block_size: int = 1024,
    block_idx_first_scheduled_token: torch.Tensor | None = None,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None
) -> None:
    """Wrapper for vLLM selective_scan_fwd implementation."""
    _custom_ops.selective_scan_fwd(
        u,
        delta,
        A,
        B,
        C,
        D_,
        z_,
        delta_bias_,
        delta_softplus,
        query_start_loc,
        cache_indices,
        has_initial_state,
        ssm_states,
        pad_slot_id,
        block_size,
        block_idx_first_scheduled_token,
        block_idx_last_scheduled_token,
        initial_state_idx
    )
