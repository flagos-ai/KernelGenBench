"""
验证 vllm_ops_api_reference.md 中所有算子API是否可访问。
只验证函数/方法是否存在，不实际执行。

用法:
    source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench
    python scripts/verify_vllm_ops_api.py
"""
import sys
import importlib
import traceback

import torch


# ============================================================
# 第一部分：Triton 算子（31个）
# 格式: (序号, 算子名, 模块路径, 函数名)
# ============================================================
TRITON_OPS = [
    (1, "triton_reshape_and_cache_flash",
     "vllm.attention.ops.triton_reshape_and_cache_flash",
     "triton_reshape_and_cache_flash"),
    (2, "fused_recurrent_gated_delta_rule_fwd",
     "vllm.model_executor.layers.fla.ops.fused_recurrent",
     "fused_recurrent_gated_delta_rule_fwd"),
    (3, "awq_gemm_triton",
     "vllm.model_executor.layers.quantization.awq_triton",
     "awq_gemm_triton"),
    (4, "triton_scaled_mm",
     "vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm",
     "triton_scaled_mm"),
    (5, "fused_gdn_gating",
     "vllm.model_executor.models.qwen3_next",
     "fused_gdn_gating"),
    (6, "triton_convert_req_index_to_global_index",
     "vllm.v1.attention.backends.mla.flashmla_sparse",
     "triton_convert_req_index_to_global_index"),
    (7, "zero_experts_compute_triton",
     "vllm.model_executor.layers.fused_moe.fused_moe",
     "zero_experts_compute_triton"),
    (8, "linear_decode_forward_triton",
     "vllm.model_executor.layers.lightning_attn",
     "linear_decode_forward_triton"),
    (9, "awq_dequantize_triton",
     "vllm.model_executor.layers.quantization.awq_triton",
     "awq_dequantize_triton"),
    (10, "pack_seq_triton",
     "vllm.attention.ops.common",
     "pack_seq_triton"),
    (11, "triton_mrope",
     "vllm.model_executor.layers.rotary_embedding.mrope",
     "triton_mrope"),
    (12, "unpack_seq_triton",
     "vllm.attention.ops.common",
     "unpack_seq_triton"),
    (13, "chunk_scaled_dot_kkt_fwd",
     "vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt",
     "chunk_scaled_dot_kkt_fwd"),
    (14, "_layer_norm_fwd",
     "vllm.model_executor.layers.mamba.ops.layernorm_gated",
     "_layer_norm_fwd"),
    (15, "layer_norm_fwd",
     "vllm.model_executor.layers.fla.ops.layernorm_guard",
     "layer_norm_fwd"),
    (16, "_bmm_chunk_fwd",
     "vllm.model_executor.layers.mamba.ops.ssd_bmm",
     "_bmm_chunk_fwd"),
    (17, "_chunk_cumsum_fwd",
     "vllm.model_executor.layers.mamba.ops.ssd_chunk_state",
     "_chunk_cumsum_fwd"),
    (18, "_chunk_scan_fwd",
     "vllm.model_executor.layers.mamba.ops.ssd_chunk_scan",
     "_chunk_scan_fwd"),
    (19, "_chunk_state_fwd",
     "vllm.model_executor.layers.mamba.ops.ssd_chunk_state",
     "_chunk_state_fwd"),
    (20, "chunk_fwd_o",
     "vllm.model_executor.layers.fla.ops.chunk_o",
     "chunk_fwd_o"),
    (21, "chunk_gated_delta_rule_fwd_h",
     "vllm.model_executor.layers.fla.ops.chunk_delta_h",
     "chunk_gated_delta_rule_fwd_h"),
    (22, "_decode_grouped_att_m_fwd",
     "vllm.attention.ops.triton_decode_attention",
     "_decode_grouped_att_m_fwd"),
    (23, "_decode_softmax_reducev_fwd",
     "vllm.attention.ops.triton_decode_attention",
     "_decode_softmax_reducev_fwd"),
    (24, "_decode_att_m_fwd",
     "vllm.attention.ops.triton_decode_attention",
     "_decode_att_m_fwd"),
    (25, "_state_passing_fwd",
     "vllm.model_executor.layers.mamba.ops.ssd_state_passing",
     "_state_passing_fwd"),
    (26, "l2norm_fwd",
     "vllm.model_executor.layers.fla.ops.l2norm",
     "l2norm_fwd"),
    (27, "recompute_w_u_fwd",
     "vllm.model_executor.layers.fla.ops.wy_fast",
     "recompute_w_u_fwd"),
    (28, "chunk_state_varlen",
     "vllm.model_executor.layers.mamba.ops.ssd_chunk_state",
     "chunk_state_varlen"),
    (29, "chunk_local_cumsum_scalar",
     "vllm.model_executor.layers.fla.ops.cumsum",
     "chunk_local_cumsum_scalar"),
    (30, "chunk_local_cumsum_vector",
     "vllm.model_executor.layers.fla.ops.cumsum",
     "chunk_local_cumsum_vector"),
    (31, "sample_recovered_tokens",
     "vllm.v1.sample.rejection_sampler",
     "sample_recovered_tokens"),
]

# ============================================================
# 第二部分：CUDA A类算子（110个）
# 通过 from vllm import _custom_ops as ops; ops.xxx 调用
# ============================================================
CUDA_A_OPS = [
    "paged_attention_v1",
    "paged_attention_v2",
    "paged_attention_rocm",
    "mla_decode_kvcache_cpu",
    "merge_attn_states",
    "flash_mla_with_kvcache",
    "get_flash_mla_metadata",
    "sm100_cutlass_mla_decode",
    "sm100_cutlass_mla_get_workspace_size",
    "convert_vertical_slash_indexes",
    "convert_vertical_slash_indexes_mergehead",
    "reshape_and_cache",
    "reshape_and_cache_flash",
    "concat_and_cache_mla",
    "copy_blocks",
    "copy_blocks_mla",
    "swap_blocks",
    "convert_fp8",
    "gather_and_maybe_dequant_cache",
    "cp_gather_cache",
    "indexer_k_quant_and_cache",
    "rms_norm",
    "fused_add_rms_norm",
    "rms_norm_dynamic_per_token_quant",
    "rms_norm_per_block_quant",
    "rotary_embedding",
    "fused_qk_norm_rope",
    "scaled_fp8_quant",
    "scaled_fp4_quant",
    "scaled_fp4_experts_quant",
    "scaled_int8_quant",
    "cutlass_pack_scale_fp8",
    "cutlass_scaled_mm",
    "cutlass_scaled_mm_azp",
    "cutlass_scaled_fp4_mm",
    "cutlass_scaled_sparse_mm",
    "cutlass_blockwise_scaled_grouped_mm",
    "cutlass_w4a8_mm",
    "cutlass_sparse_compress",
    "awq_gemm",
    "awq_dequantize",
    "gptq_gemm",
    "gptq_shuffle",
    "gptq_marlin_24_gemm",
    "gptq_marlin_gemm",
    "machete_mm",
    "machete_prepack_B",
    "machete_supported_schedules",
    "allspark_w8a16_gemm",
    "allspark_repack_weight",
    "LLMM1",
    "wvSplitK",
    "wvSplitKQ",
    # Repack
    "gptq_marlin_repack",
    "awq_marlin_repack",
    "gptq_marlin_moe_repack",
    "awq_marlin_moe_repack",
    "cutlass_encode_and_reorder_int4b",
    "permute_cols",
    "shuffle_rows",
    # MoE
    "moe_align_block_size",
    "moe_sum",
    "topk_softmax",
    "grouped_topk",
    "cutlass_moe_mm",
    "cutlass_fp4_moe_mm",
    "get_cutlass_moe_mm_data",
    "get_cutlass_moe_mm_problem_sizes",
    "get_cutlass_pplx_moe_mm_data",
    "moe_wna16_gemm",
    "moe_wna16_marlin_gemm",
    # GGML
    "ggml_dequantize",
    "ggml_mul_mat_vec_a8",
    "ggml_mul_mat_a8",
    "ggml_moe_a8",
    "ggml_moe_a8_vec",
    "ggml_moe_get_block_size",
    # SSM/Mamba
    "selective_scan_fwd",
    # Sampling
    "apply_repetition_penalties",
    "apply_repetition_penalties_cuda",
    "apply_repetition_penalties_torch",
    # Device/Capability
    "get_device_attribute",
    "get_max_shared_memory_per_block_device_attribute",
    "cutlass_scaled_mm_supports_fp8",
    "cutlass_scaled_mm_supports_fp4",
    "cutlass_scaled_mm_supports_block_fp8",
    "cutlass_sparse_scaled_mm_supported",
    "cutlass_group_gemm_supported",
    # Communication/AllReduce
    "all_reduce",
    "init_custom_ar",
    "dispose",
    "meta_size",
    "register_buffer",
    "get_graph_buffer_ipc_meta",
    "register_graph_buffers",
    "allocate_shared_buffer_and_handle",
    "open_mem_handle",
    "free_shared_buffer",
    "qr_all_reduce",
    "qr_get_handle",
    "qr_open_handles",
    "qr_max_size",
    "qr_destroy",
    # Other
    "hadacore_transform",
]

# ============================================================
# 第二部分：CUDA B类算子（25个）
# 直接通过 torch.ops._C / _moe_C 调用
# 格式: (序号, 算子名, 命名空间, 函数名)
# ============================================================
CUDA_B_OPS = [
    # Activation
    (111, "silu_and_mul", "_C", "silu_and_mul"),
    (112, "gelu_and_mul", "_C", "gelu_and_mul"),
    (113, "gelu_tanh_and_mul", "_C", "gelu_tanh_and_mul"),
    (114, "mul_and_silu", "_C", "mul_and_silu"),
    (115, "fatrelu_and_mul", "_C", "fatrelu_and_mul"),
    (116, "swigluoai_and_mul", "_C", "swigluoai_and_mul"),
    (117, "gelu_new", "_C", "gelu_new"),
    (118, "gelu_fast", "_C", "gelu_fast"),
    (119, "gelu_quick", "_C", "gelu_quick"),
    # Fused Activation + Quant
    (120, "silu_and_mul_quant", "_C", "silu_and_mul_quant"),
    (121, "silu_and_mul_nvfp4_quant", "_C", "silu_and_mul_nvfp4_quant"),
    # Fused Norm + Quant
    (122, "fused_add_rms_norm_static_fp8_quant", "_C",
     "fused_add_rms_norm_static_fp8_quant"),
    (123, "rms_norm_static_fp8_quant", "_C", "rms_norm_static_fp8_quant"),
    # Low-level Quantization
    (124, "dynamic_per_token_scaled_fp8_quant", "_C",
     "dynamic_per_token_scaled_fp8_quant"),
    (125, "dynamic_scaled_fp8_quant", "_C", "dynamic_scaled_fp8_quant"),
    (126, "static_scaled_fp8_quant", "_C", "static_scaled_fp8_quant"),
    (127, "static_scaled_int8_quant", "_C", "static_scaled_int8_quant"),
    (128, "dynamic_scaled_int8_quant", "_C", "dynamic_scaled_int8_quant"),
    # Other Low-level
    (129, "per_token_group_quant_int8", "_C", "per_token_group_quant_int8"),
    (130, "top_k_per_row_prefill", "_C", "top_k_per_row_prefill"),
    (131, "top_k_per_row_decode", "_C", "top_k_per_row_decode"),
    (132, "apply_repetition_penalties_", "_C", "apply_repetition_penalties_"),
    (133, "persistent_masked_m_silu_mul_quant", "_C",
     "persistent_masked_m_silu_mul_quant"),
    # 其他命名空间
    (134, "cutlass_fp4_group_mm", "_C", "cutlass_fp4_group_mm"),
    (135, "vllm_topk_softmax", "_moe_C", "topk_softmax"),
]


# ============================================================
# 验证函数
# ============================================================
def verify_triton_ops():
    """验证Triton算子：import模块并检查函数是否存在"""
    results = []
    for idx, name, module_path, func_name in TRITON_OPS:
        try:
            mod = importlib.import_module(module_path)
            fn = getattr(mod, func_name)
            if callable(fn):
                results.append((idx, name, "PASS", ""))
            else:
                results.append((idx, name, "FAIL",
                                f"{func_name} 存在但不可调用"))
        except Exception as e:
            results.append((idx, name, "FAIL", str(e)[:120]))
    return results


def verify_cuda_a_ops(ops_module):
    """验证CUDA A类算子：检查 ops.xxx 是否存在"""
    results = []
    for idx, name in enumerate(CUDA_A_OPS, start=1):
        try:
            fn = getattr(ops_module, name)
            if callable(fn):
                results.append((idx, name, "PASS", ""))
            else:
                results.append((idx, name, "FAIL",
                                f"ops.{name} 存在但不可调用"))
        except AttributeError:
            results.append((idx, name, "FAIL",
                            f"ops.{name} 不存在"))
        except Exception as e:
            results.append((idx, name, "FAIL", str(e)[:120]))
    return results


def verify_cuda_b_ops():
    """验证CUDA B类算子：检查 torch.ops.<namespace>.<func> 是否存在"""
    results = []
    for idx, name, namespace, func_name in CUDA_B_OPS:
        try:
            ns = getattr(torch.ops, namespace)
            fn = getattr(ns, func_name)
            if callable(fn):
                results.append((idx, name, "PASS", ""))
            else:
                results.append((idx, name, "FAIL",
                                f"torch.ops.{namespace}.{func_name} 存在但不可调用"))
        except AttributeError:
            results.append((idx, name, "FAIL",
                            f"torch.ops.{namespace}.{func_name} 不存在"))
        except Exception as e:
            results.append((idx, name, "FAIL", str(e)[:120]))
    return results


# ============================================================
# 输出与主函数
# ============================================================
def print_results(title, results):
    """打印验证结果表格"""
    passed = sum(1 for r in results if r[2] == "PASS")
    failed = sum(1 for r in results if r[2] == "FAIL")
    total = len(results)

    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"  PASS: {passed}/{total}  FAIL: {failed}/{total}")
    print(f"{'=' * 60}")

    for idx, name, status, msg in results:
        mark = "✓" if status == "PASS" else "✗"
        line = f"  [{mark}] {idx:>3}. {name}"
        if msg:
            line += f"  -- {msg}"
        print(line)


def main():
    print("=" * 60)
    print("  vLLM v0.13 算子API可访问性验证")
    print("=" * 60)

    # ---------- Triton ----------
    triton_results = verify_triton_ops()
    print_results("第一部分：Triton 算子（31个）", triton_results)

    # ---------- CUDA A ----------
    try:
        from vllm import _custom_ops as ops
        cuda_a_results = verify_cuda_a_ops(ops)
    except Exception as e:
        print(f"\n[ERROR] 无法导入 vllm._custom_ops: {e}")
        cuda_a_results = [(i, n, "FAIL", "无法导入 _custom_ops")
                          for i, n in enumerate(CUDA_A_OPS, start=1)]
    print_results("第二部分A：CUDA A类算子（104个, _custom_ops）",
                  cuda_a_results)

    # ---------- CUDA B ----------
    cuda_b_results = verify_cuda_b_ops()
    print_results("第二部分B：CUDA B类算子（25个, torch.ops直接调用）",
                  cuda_b_results)

    # ---------- 汇总 ----------
    all_results = triton_results + cuda_a_results + cuda_b_results
    total = len(all_results)
    passed = sum(1 for r in all_results if r[2] == "PASS")
    failed = total - passed

    print(f"\n{'=' * 60}")
    print(f"  总计: {total} 个算子")
    print(f"  PASS: {passed}  FAIL: {failed}")
    print(f"  通过率: {passed/total*100:.1f}%")
    print(f"{'=' * 60}")

    if failed > 0:
        print(f"\n失败算子列表:")
        for idx, name, status, msg in all_results:
            if status == "FAIL":
                print(f"  - {name}: {msg}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
