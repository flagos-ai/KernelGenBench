# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# vLLM13 Baseline Functions
from .allspark_w8a16_gemm import allspark_w8a16_gemm
from .apply_repetition_penalties_cuda import apply_repetition_penalties_cuda
from .awq_gemm import awq_gemm
from .awq_marlin_moe_repack import awq_marlin_moe_repack
from .batched_moe_align_block_size import batched_moe_align_block_size
from .concat_and_cache_mla import concat_and_cache_mla
from .convert_fp8 import convert_fp8
from .convert_vertical_slash_indexes import convert_vertical_slash_indexes
from .copy_blocks import copy_blocks
from .copy_blocks_mla import copy_blocks_mla
from .cp_gather_cache import cp_gather_cache
from .cp_gather_indexer_k_quant_cache import cp_gather_indexer_k_quant_cache
from .cutlass_pack_scale_fp8 import cutlass_pack_scale_fp8
from .cutlass_scaled_mm import cutlass_scaled_mm
from .cutlass_scaled_mm_azp import cutlass_scaled_mm_azp
from .fused_add_rms_norm import fused_add_rms_norm
from .fused_qk_norm_rope import fused_qk_norm_rope
from .gather_and_maybe_dequant_cache import gather_and_maybe_dequant_cache
from .ggml_dequantize import ggml_dequantize
from .ggml_moe_a8 import ggml_moe_a8
from .ggml_moe_a8_vec import ggml_moe_a8_vec
from .ggml_mul_mat_a8 import ggml_mul_mat_a8
from .ggml_mul_mat_vec_a8 import ggml_mul_mat_vec_a8
from .gptq_gemm import gptq_gemm
from .gptq_marlin_24_gemm import gptq_marlin_24_gemm
from .gptq_marlin_gemm import gptq_marlin_gemm
from .gptq_marlin_moe_repack import gptq_marlin_moe_repack
from .gptq_shuffle import gptq_shuffle
from .grouped_topk import grouped_topk
from .hadacore_transform import hadacore_transform
from .marlin_int4_fp8_preprocess import marlin_int4_fp8_preprocess
from .merge_attn_states import merge_attn_states
from .moe_align_block_size import moe_align_block_size
from .moe_lora_align_block_size import moe_lora_align_block_size
from .moe_sum import moe_sum
from .paged_attention_v1 import paged_attention_v1
from .paged_attention_v2 import paged_attention_v2
from .permute_cols import permute_cols
from .reshape_and_cache import reshape_and_cache
from .reshape_and_cache_flash import reshape_and_cache_flash
from .rms_norm import rms_norm
from .rms_norm_dynamic_per_token_quant import rms_norm_dynamic_per_token_quant
from .rms_norm_per_block_quant import rms_norm_per_block_quant
from .rotary_embedding import rotary_embedding
from .scaled_fp8_quant import scaled_fp8_quant
from .scaled_int8_quant import scaled_int8_quant
from .selective_scan_fwd import selective_scan_fwd
from .shuffle_rows import shuffle_rows
from .swap_blocks import swap_blocks
from .topk_softmax import topk_softmax
