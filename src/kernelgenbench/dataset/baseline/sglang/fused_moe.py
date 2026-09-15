"""
SGLang fused_moe baseline (Triton fused MoE).
"""
import torch
import typing
try:
    from sglang.srt.layers.moe.fused_moe_triton.triton_kernels_moe import triton_kernel_fused_experts as _fused_experts
except ModuleNotFoundError:
    _fused_experts = None



def fused_moe(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, routing_data: "typing.Any", gather_indx: "typing.Any", scatter_indx: "typing.Any", inplace: bool = False, activation: str = 'silu', apply_router_weight_on_input: bool = False) -> torch.Tensor:
    """
    Args:
        hidden_states: torch.Tensor
        w1: torch.Tensor
        w2: torch.Tensor
        routing_data: "typing.Any"
        gather_indx: "typing.Any"
        scatter_indx: "typing.Any"
        inplace: bool
        activation: str
        apply_router_weight_on_input: bool
    Returns:
        torch.Tensor
    """
    return _fused_experts(hidden_states, w1, w2, routing_data, gather_indx, scatter_indx, inplace=inplace, activation=activation, apply_router_weight_on_input=apply_router_weight_on_input)

