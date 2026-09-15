"""
SGLang baseline operators.

Each submodule wraps a single SGLang operator as a callable function.
Follows the vLLM baseline pattern: thin Python wrappers with try/except imports.

Categories mirror SGLang source modules:
  layers/activation.py, layers/layernorm.py, layers/rotary_embedding/,
  layers/moe/, layers/attention/fla/, layers/attention/mamba/,
  layers/elementwise.py, layers/gemma4_fused_ops.py,
  layers/conv.py, layers/quantization/
"""

# layers/activation.py
from .silu_and_mul import silu_and_mul
from .gelu_and_mul import gelu_and_mul
from .quick_gelu import quick_gelu
from .new_gelu import new_gelu
from .xielu import xielu

# layers/layernorm.py
from .rms_norm import rms_norm
from .layer_norm import layer_norm
from .gemma_rms_norm import gemma_rms_norm
from .gemma3_rms_norm import gemma3_rms_norm
from .gemma4_rms_norm import gemma4_rms_norm
from .rms_norm_without_scale import rms_norm_without_scale

# layers/rotary_embedding/
from .rotary_embedding import rotary_embedding
from .mrotary_embedding import mrotary_embedding
from .dual_chunk_rotary_embedding import dual_chunk_rotary_embedding
from .deepseek_scaling_rotary_embedding import deepseek_scaling_rotary_embedding
from .llama3_rotary_embedding import llama3_rotary_embedding
from .dynamic_ntk_scaling_rotary_embedding import dynamic_ntk_scaling_rotary_embedding
from .linear_scaling_rotary_embedding import linear_scaling_rotary_embedding
from .phi3_long_rope_scaled_rotary_embedding import phi3_long_rope_scaled_rotary_embedding
from .triton_mrope_fused import triton_mrope_fused
from .triton_ernie45_rope_fused import triton_ernie45_rope_fused
from .apply_interleaved_rope_triton import apply_interleaved_rope_triton
from .dynamic_ntk_alpha_rotary_embedding import dynamic_ntk_alpha_rotary_embedding

# layers/moe/
from .fused_moe import fused_moe
from .topk import topk
from .moe_align_block_size import moe_align_block_size

# layers/attention/fla/
from .l2norm import l2norm
from .rms_norm_gated import rms_norm_gated
from .fused_recurrent_gated_delta_rule import fused_recurrent_gated_delta_rule
from .fused_recurrent_gated_delta_rule_update import fused_recurrent_gated_delta_rule_update
from .fused_sigmoid_gating_delta_rule_update import fused_sigmoid_gating_delta_rule_update
from .fused_sigmoid_gating_delta_rule_packed_decode import fused_sigmoid_gating_delta_rule_packed_decode
from .fused_gdn_gating import fused_gdn_gating
from .layer_norm_gated_fwd import layer_norm_gated_fwd

# layers/attention/mamba/
from .causal_conv1d_fn import causal_conv1d_fn
from .causal_conv1d_update import causal_conv1d_update
from .selective_scan_update import selective_scan_update
from .mamba_chunk_scan_combined_fwd import mamba_chunk_scan_combined_fwd
from .mixer2_rms_norm_gated import mixer2_rms_norm_gated

# layers/elementwise.py
from .fused_dual_residual_rmsnorm import fused_dual_residual_rmsnorm
from .softcap import softcap
from .silu_and_mul_triton import silu_and_mul_triton
from .gelu_and_mul_triton import gelu_and_mul_triton
from .fused_rmsnorm import fused_rmsnorm
from .experts_combine_triton import experts_combine_triton

# layers/gemma4_fused_ops.py
from .gemma_rmsnorm_residual_scalar import gemma_rmsnorm_residual_scalar
from .gemma_qkv_rmsnorm import gemma_qkv_rmsnorm

# layers/conv.py
from .conv2d_layer import conv2d_layer
from .conv3d_layer import conv3d_layer

# layers/quantization/
from .per_token_quant_int8 import per_token_quant_int8

__all__ = [
    # layers/activation.py
    "silu_and_mul", "gelu_and_mul", "quick_gelu", "new_gelu", "xielu",
    # layers/layernorm.py
    "rms_norm", "layer_norm", "gemma_rms_norm", "gemma3_rms_norm",
    "gemma4_rms_norm", "rms_norm_without_scale",
    # layers/rotary_embedding/
    "rotary_embedding", "mrotary_embedding", "dual_chunk_rotary_embedding",
    "deepseek_scaling_rotary_embedding", "llama3_rotary_embedding",
    "dynamic_ntk_scaling_rotary_embedding", "linear_scaling_rotary_embedding",
    "phi3_long_rope_scaled_rotary_embedding",
    "triton_mrope_fused", "triton_ernie45_rope_fused",
    "apply_interleaved_rope_triton", "dynamic_ntk_alpha_rotary_embedding",
    # layers/moe/
    "fused_moe", "topk", "moe_align_block_size",
    # layers/attention/fla/
    "l2norm", "rms_norm_gated",
    "fused_recurrent_gated_delta_rule", "fused_recurrent_gated_delta_rule_update",
    "fused_sigmoid_gating_delta_rule_update",
    "fused_sigmoid_gating_delta_rule_packed_decode",
    "fused_gdn_gating", "layer_norm_gated_fwd",
    # layers/attention/mamba/
    "causal_conv1d_fn", "causal_conv1d_update",
    "selective_scan_update", "mamba_chunk_scan_combined_fwd",
    "mixer2_rms_norm_gated",
    # layers/elementwise.py
    "fused_dual_residual_rmsnorm",
    "softcap", "silu_and_mul_triton", "gelu_and_mul_triton",
    "fused_rmsnorm", "experts_combine_triton",
    # layers/gemma4_fused_ops.py
    "gemma_rmsnorm_residual_scalar", "gemma_qkv_rmsnorm",
    # layers/conv.py
    "conv2d_layer", "conv3d_layer",
    # layers/quantization/
    "per_token_quant_int8",
]
