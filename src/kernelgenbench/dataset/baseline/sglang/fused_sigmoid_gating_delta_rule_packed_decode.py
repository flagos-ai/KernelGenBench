"""
SGLang fused_sigmoid_gating_delta_rule_packed_decode baseline.
"""
import torch
import typing
try:
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_packed_decode as _fused_sig_rule_packed
except ModuleNotFoundError:
    _fused_sig_rule_packed = None



def fused_sigmoid_gating_delta_rule_packed_decode(mixed_qkv: torch.Tensor, a: torch.Tensor, b: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor, scale: float, initial_state: torch.Tensor, out: torch.Tensor, ssm_state_indices: torch.Tensor, use_qk_l2norm_in_kernel: bool = False) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        mixed_qkv: torch.Tensor
        a: torch.Tensor
        b: torch.Tensor
        A_log: torch.Tensor
        dt_bias: torch.Tensor
        scale: float
        initial_state: torch.Tensor
        out: torch.Tensor
        ssm_state_indices: torch.Tensor
        use_qk_l2norm_in_kernel: bool
    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]
    """
    return _fused_sig_rule_packed(mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=scale, initial_state=initial_state, out=out, ssm_state_indices=ssm_state_indices, use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel)

