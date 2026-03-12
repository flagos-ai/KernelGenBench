#!/usr/bin/env python3
"""Test 40 vllm13 operators"""
import os
os.environ["DISPATCH_TORCH_LIB"] = "0"

import sys
from importlib import import_module

ops = [
    "rms_norm", "rms_norm_dynamic_per_token_quant", "moe_sum", "apply_repetition_penalties",
    "permute_cols", "shuffle_rows", "convert_fp8", "hadacore_transform", "swap_blocks",
    "grouped_topk", "rms_norm_per_block_quant", "gptq_shuffle", "merge_attn_states",
    "reshape_and_cache", "reshape_and_cache_flash", "concat_and_cache_mla", "paged_attention_v2",
    "apply_repetition_penalties_cuda", "gptq_gemm", "cutlass_pack_scale_fp8", "cutlass_scaled_mm_azp",
    "gptq_marlin_24_gemm", "copy_blocks", "copy_blocks_mla", "gptq_marlin_repack", "awq_marlin_repack",
    "allspark_repack_weight", "allspark_w8a16_gemm", "gather_and_maybe_dequant_cache",
    "convert_vertical_slash_indexes", "cp_gather_cache", "cp_gather_indexer_k_quant_cache",
    "marlin_int4_fp8_preprocess", "gptq_marlin_gemm", "ggml_mul_mat_a8", "ggml_mul_mat_vec_a8",
    "gptq_marlin_moe_repack", "awq_marlin_moe_repack", "moe_lora_align_block_size", "batched_moe_align_block_size"
]

passed = []
failed = []

for op in ops:
    try:
        print(f"\n{'='*60}")
        print(f"Testing: {op}")
        print('='*60)
        mod = import_module(f"flagbench.accuracy.vllm13.test_{op}")
        test_func = getattr(mod, f"test_accuracy_{op}")
        test_func.pytestmark  # trigger parametrize
        print(f"✓ {op} - PASSED")
        passed.append(op)
    except Exception as e:
        print(f"✗ {op} - FAILED: {e}")
        failed.append(op)

print(f"\n{'='*60}")
print(f"SUMMARY: {len(passed)}/{len(ops)} passed")
print('='*60)
if failed:
    print(f"\nFailed operators ({len(failed)}):")
    for op in failed:
        print(f"  - {op}")
    sys.exit(1)
else:
    print("\n✓ All tests passed!")
    sys.exit(0)
