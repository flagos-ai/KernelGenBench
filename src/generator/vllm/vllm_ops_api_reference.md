# vLLM v0.13 算子API调用参考

> 统计：可运行 160 个（Triton 31 + CUDA 129），不可运行 4 个，合计 164 个 GPU 算子

## 第一部分：Triton 可运行算子（31个）

> 调用方式：直接 import wrapper 函数，传入 tensor 参数调用
> 这些函数内部实现了 @triton.jit kernel，是 triton kernel 的对外接口

### 1. triton_reshape_and_cache_flash
```python
from vllm.attention.ops.triton_reshape_and_cache_flash import triton_reshape_and_cache_flash
triton_reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)
```

### 2. fused_recurrent_gated_delta_rule_fwd
```python
from vllm.model_executor.layers.fla.ops.fused_recurrent import fused_recurrent_gated_delta_rule_fwd
fused_recurrent_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state, ...)
```

### 3. awq_gemm_triton
```python
from vllm.model_executor.layers.quantization.awq_triton import awq_gemm_triton
awq_gemm_triton(input, qweight, scales, qzeros, split_k_iters, block_size_m=32, block_size_n=32, block_size_k=32)
```

### 4. triton_scaled_mm
```python
from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import triton_scaled_mm
triton_scaled_mm(input, weight, scale_a, scale_b, out_dtype, bias=None, ...)
```

### 5. fused_gdn_gating
```python
from vllm.model_executor.models.qwen3_next import fused_gdn_gating
fused_gdn_gating(A_log, a, b, dt_bias, beta=1.0, threshold=20.0)
```

### 6. triton_convert_req_index_to_global_index
```python
from vllm.v1.attention.backends.mla.flashmla_sparse import triton_convert_req_index_to_global_index
triton_convert_req_index_to_global_index(req_id, block_table, token_indices, ...)
```

### 7. zero_experts_compute_triton
```python
from vllm.model_executor.layers.fused_moe.fused_moe import zero_experts_compute_triton
zero_experts_compute_triton(...)
```

### 8. linear_decode_forward_triton
```python
from vllm.model_executor.layers.lightning_attn import linear_decode_forward_triton
linear_decode_forward_triton(...)
```

### 9. awq_dequantize_triton
```python
from vllm.model_executor.layers.quantization.awq_triton import awq_dequantize_triton
awq_dequantize_triton(qweight, scales, zeros, block_size_x=32, block_size_y=32)
```

### 10. pack_seq_triton
```python
from vllm.attention.ops.common import pack_seq_triton
pack_seq_triton(...)
```

### 11. triton_mrope
```python
from vllm.model_executor.layers.rotary_embedding.mrope import triton_mrope
triton_mrope(q, k, cos, sin, mrope_section, head_size, rotary_dim, mrope_interleaved)
```

### 12. unpack_seq_triton
```python
from vllm.attention.ops.common import unpack_seq_triton
unpack_seq_triton(...)
```

### 13. chunk_scaled_dot_kkt_fwd
```python
from vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
chunk_scaled_dot_kkt_fwd(...)
```

### 14. _layer_norm_fwd
```python
from vllm.model_executor.layers.mamba.ops.layernorm_gated import _layer_norm_fwd
_layer_norm_fwd(x, weight, bias, eps, z=None, out=None, group_size=None, norm_before_gate=True, is_rms_norm=False)
```

### 15. layer_norm_fwd
```python
from vllm.model_executor.layers.fla.ops.layernorm_guard import layer_norm_fwd
layer_norm_fwd(...)
```

### 16. _bmm_chunk_fwd
```python
from vllm.model_executor.layers.mamba.ops.ssd_bmm import _bmm_chunk_fwd
_bmm_chunk_fwd(...)
```

### 17. _chunk_cumsum_fwd
```python
from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import _chunk_cumsum_fwd
_chunk_cumsum_fwd(...)
```

### 18. _chunk_scan_fwd
```python
from vllm.model_executor.layers.mamba.ops.ssd_chunk_scan import _chunk_scan_fwd
_chunk_scan_fwd(...)
```

### 19. _chunk_state_fwd
```python
from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import _chunk_state_fwd
_chunk_state_fwd(...)
```

### 20. chunk_fwd_o
```python
from vllm.model_executor.layers.fla.ops.chunk_o import chunk_fwd_o
chunk_fwd_o(...)
```

### 21. chunk_gated_delta_rule_fwd_h
```python
from vllm.model_executor.layers.fla.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h
chunk_gated_delta_rule_fwd_h(...)
```

### 22. _decode_grouped_att_m_fwd
```python
from vllm.attention.ops.triton_decode_attention import _decode_grouped_att_m_fwd
_decode_grouped_att_m_fwd(q, k_buffer, v_buffer, att_out, Req_to_tokens, B_Seqlen, num_kv_splits, sm_scale, page_size, logit_cap)
```

### 23. _decode_softmax_reducev_fwd
```python
from vllm.attention.ops.triton_decode_attention import _decode_softmax_reducev_fwd
_decode_softmax_reducev_fwd(logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits)
```

### 24. _decode_att_m_fwd
```python
from vllm.attention.ops.triton_decode_attention import _decode_att_m_fwd
_decode_att_m_fwd(q, k_buffer, v_buffer, att_out, Req_to_tokens, B_Seqlen, num_kv_splits, sm_scale, page_size, logit_cap)
```

### 25. _state_passing_fwd
```python
from vllm.model_executor.layers.mamba.ops.ssd_state_passing import _state_passing_fwd
_state_passing_fwd(...)
```

### 26. l2norm_fwd
```python
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
l2norm_fwd(x, eps=1e-6, output_dtype=None)
```

### 27. recompute_w_u_fwd
```python
from vllm.model_executor.layers.fla.ops.wy_fast import recompute_w_u_fwd
recompute_w_u_fwd(k, v, beta, g_cumsum, A, cu_seqlens)
```

### 28. chunk_state_varlen
```python
from vllm.model_executor.layers.mamba.ops.ssd_chunk_state import chunk_state_varlen
chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states, initial_states=None)
```

### 29. chunk_local_cumsum_scalar
```python
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum_scalar
chunk_local_cumsum_scalar(...)
```

### 30. chunk_local_cumsum_vector
```python
from vllm.model_executor.layers.fla.ops.cumsum import chunk_local_cumsum_vector
chunk_local_cumsum_vector(...)
```

### 31. sample_recovered_tokens
```python
from vllm.v1.sample.rejection_sampler import sample_recovered_tokens
sample_recovered_tokens(...)
```

---

## 第二部分：CUDA 可运行算子（129个）

> 调用方式分两类：
> - **A类**：通过 `_custom_ops.py` 封装，调用方式为 `from vllm import _custom_ops as ops; ops.xxx(...)`
> - **B类**：无 `_custom_ops.py` 封装，直接通过 `torch.ops._C.xxx(...)` 调用

### A类：通过 _custom_ops.py 调用（104个）

> ```python
> from vllm import _custom_ops as ops
> ops.xxx(...)
> ```

#### Attention 算子

**1. paged_attention_v1**
```python
ops.paged_attention_v1(out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype, k_scale, v_scale, tp_rank=0, blocksparse_local_blocks=0, blocksparse_vert_stride=0, blocksparse_block_size=64, blocksparse_head_sliding_step=0)
```

**2. paged_attention_v2**
```python
ops.paged_attention_v2(out, exp_sum, max_logits, tmp_out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, seq_lens, block_size, max_seq_len, alibi_slopes, kv_cache_dtype, k_scale, v_scale, tp_rank=0, blocksparse_local_blocks=0, blocksparse_vert_stride=0, blocksparse_block_size=64, blocksparse_head_sliding_step=0)
```

**3. paged_attention_rocm**
```python
ops.paged_attention_rocm(out, exp_sum, max_logits, tmp_out, query, key_cache, value_cache, num_kv_heads, scale, block_tables, seq_lens, query_start_loc, block_size, max_seq_len, alibi_slopes, kv_cache_dtype, k_scale, v_scale, fp8_out_scale=None, mfma_type="f16")
```

**4. mla_decode_kvcache_cpu**
```python
ops.mla_decode_kvcache_cpu(out, query, kv_cache, scale, block_tables, seq_lens)
```

**5. merge_attn_states**
```python
ops.merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse=None)
```

**6. flash_mla_with_kvcache**
```python
ops.flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens, head_dim_v, tile_scheduler_metadata, num_splits, softmax_scale=None, causal=False)
```

**7. get_flash_mla_metadata**
```python
ops.get_flash_mla_metadata(cache_seqlens, num_heads_per_head_k, num_heads_k)
```

**8. sm100_cutlass_mla_decode**
```python
ops.sm100_cutlass_mla_decode(out, lse, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, workspace, scale, num_kv_splits)
```

**9. sm100_cutlass_mla_get_workspace_size**
```python
ops.sm100_cutlass_mla_get_workspace_size(max_seq_len, num_batches, sm_count, num_kv_splits)
```

**10. convert_vertical_slash_indexes**
```python
ops.convert_vertical_slash_indexes(q_seqlens, kv_seqlens, vertical_indexes, slash_indexes, context_size, block_size_M, block_size_N, causal=True)
```

**11. convert_vertical_slash_indexes_mergehead**
```python
ops.convert_vertical_slash_indexes_mergehead(q_seqlens, kv_seqlens, vertical_indexes, slash_indexes, vertical_indices_count, slash_indices_count, context_size, block_size_M, block_size_N, causal=True)
```

#### KV Cache 算子

**12. reshape_and_cache**
```python
ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)
```

**13. reshape_and_cache_flash**
```python
ops.reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)
```

**14. concat_and_cache_mla**
```python
ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale)
```

**15. copy_blocks**
```python
ops.copy_blocks(key_caches, value_caches, block_mapping)
```

**16. copy_blocks_mla**
```python
ops.copy_blocks_mla(kv_caches, block_mapping)
```

**17. swap_blocks**
```python
ops.swap_blocks(src, dst, block_mapping)
```

**18. convert_fp8**
```python
ops.convert_fp8(output, input, scale=1.0, kv_dtype="fp8")
```

**19. gather_and_maybe_dequant_cache**
```python
ops.gather_and_maybe_dequant_cache(src_cache, dst, block_table, cu_seq_lens, token_to_seq, num_tokens, kv_cache_dtype, scale, seq_starts=None)
```

**20. cp_gather_cache**
```python
ops.cp_gather_cache(src_cache, dst, block_table, cu_seq_lens, batch_size, seq_starts=None)
```

**21. indexer_k_quant_and_cache**
```python
ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping, quant_block_size, kv_cache_dtype)
```

#### Norm 算子

**22. rms_norm**
```python
ops.rms_norm(out, input, weight, epsilon)
```

**23. fused_add_rms_norm**
```python
ops.fused_add_rms_norm(input, residual, weight, epsilon)
```

**24. rms_norm_dynamic_per_token_quant**
```python
ops.rms_norm_dynamic_per_token_quant(input, weight, epsilon, quant_dtype, scale_ub=None, residual=None)
```

**25. rms_norm_per_block_quant**
```python
ops.rms_norm_per_block_quant(input, weight, epsilon, quant_dtype, group_size, scale_ub=None, residual=None, is_scale_transposed=False)
```

#### Rotary Embedding 算子

**26. rotary_embedding**
```python
ops.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)
```

**27. fused_qk_norm_rope**
```python
ops.fused_qk_norm_rope(qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps, q_weight, k_weight, cos_sin_cache, is_neox, position_ids)
```

#### Quantization 算子

**28. scaled_fp8_quant**
```python
ops.scaled_fp8_quant(input, scale=None, num_token_padding=None, scale_ub=None, use_per_token_if_dynamic=False, output=None)
```

**29. scaled_fp4_quant**
```python
ops.scaled_fp4_quant(input, input_global_scale)
```

**30. scaled_fp4_experts_quant**
```python
ops.scaled_fp4_experts_quant(input_tensor, input_global_scale, expert_offsets, blockscale_offsets, topk)
```

**31. scaled_int8_quant**
```python
ops.scaled_int8_quant(input, scale=None, azp=None, symmetric=True)
```

**32. cutlass_pack_scale_fp8**
```python
ops.cutlass_pack_scale_fp8(scales)
```

#### GEMM / MatMul 算子

**33. cutlass_scaled_mm**
```python
ops.cutlass_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias=None)
```

**34. cutlass_scaled_mm_azp**
```python
ops.cutlass_scaled_mm_azp(a, b, scale_a, scale_b, out_dtype, azp_adj, azp=None, bias=None)
```

**35. cutlass_scaled_fp4_mm**
```python
ops.cutlass_scaled_fp4_mm(a, b, block_scale_a, block_scale_b, alpha, out_dtype)
```

**36. cutlass_scaled_sparse_mm**
```python
ops.cutlass_scaled_sparse_mm(a, bt_nzs, bt_meta, scale_a, scale_b, out_dtype, bias=None)
```

**37. cutlass_blockwise_scaled_grouped_mm**
```python
ops.cutlass_blockwise_scaled_grouped_mm(output, a, b, scales_a, scales_b, problem_sizes, expert_offsets)
```

**38. cutlass_w4a8_mm**
```python
ops.cutlass_w4a8_mm(a, b_q, b_group_scales, b_group_size, b_channel_scales, a_token_scales, out_type=None, maybe_schedule=None)
```

**39. cutlass_sparse_compress**
```python
ops.cutlass_sparse_compress(a)
```

**40. awq_gemm**
```python
ops.awq_gemm(input, qweight, scales, qzeros, split_k_iters)
```

**41. awq_dequantize**
```python
ops.awq_dequantize(qweight, scales, zeros, split_k_iters, thx, thy)
```

**42. gptq_gemm**
```python
ops.gptq_gemm(a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_exllama, use_v2_format, bit)
```

**43. gptq_shuffle**
```python
ops.gptq_shuffle(q_weight, q_perm, bit)
```

**44. gptq_marlin_24_gemm**
```python
ops.gptq_marlin_24_gemm(a, b_q_weight, b_meta, b_scales, workspace, b_q_type, size_m, size_n, size_k)
```

**45. gptq_marlin_gemm**
```python
ops.gptq_marlin_gemm(a, c, b_q_weight, b_bias, b_scales, a_scales, global_scale, b_zeros, g_idx, perm, workspace, b_q_type, size_m, size_n, size_k, is_k_full=True, use_atomic_add=False, use_fp32_reduce=False, is_zp_float=False)
```

**46. machete_mm**
```python
ops.machete_mm(a, b_q, b_type, out_type=None, b_group_scales=None, b_group_zeros=None, b_group_size=None, b_channel_scales=None, a_token_scales=None, schedule=None)
```

**47. machete_prepack_B**
```python
ops.machete_prepack_B(b_q_weight, a_type, b_type, group_scales_type)
```

**48. machete_supported_schedules**
```python
ops.machete_supported_schedules(a_type, b_type, group_scales_type, group_zeros_type=None, channel_scales_type=None, token_scales_type=None, out_type=None)
```

**49. allspark_w8a16_gemm**
```python
ops.allspark_w8a16_gemm(a, b_qweight, b_scales, b_qzeros, n, group_size, sm_count, sm_version, CUBLAS_M_THRESHOLD, has_zp, n32k16_reorder)
```

**50. allspark_repack_weight**
```python
ops.allspark_repack_weight(qweight, scale, zero_point=None, has_zp=False)
```

**51. LLMM1**
```python
ops.LLMM1(a, b, rows_per_block)
```

**52. wvSplitK**
```python
ops.wvSplitK(a, b, cu_count, bias=None)
```

**53. wvSplitKQ**
```python
ops.wvSplitKQ(a, b, out_dtype, scale_a, scale_b, cu_count, bias=None)
```

#### Repack 算子

**54. gptq_marlin_repack**
```python
ops.gptq_marlin_repack(b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit=False)
```

**55. awq_marlin_repack**
```python
ops.awq_marlin_repack(b_q_weight, size_k, size_n, num_bits, is_a_8bit=False)
```

**56. gptq_marlin_moe_repack**
```python
ops.gptq_marlin_moe_repack(b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit=False)
```

**57. awq_marlin_moe_repack**
```python
ops.awq_marlin_moe_repack(b_q_weight, perm, size_k, size_n, num_bits, is_a_8bit=False)
```

**58. cutlass_encode_and_reorder_int4b**
```python
ops.cutlass_encode_and_reorder_int4b(b)
```

**59. permute_cols**
```python
ops.permute_cols(a, perm)
```

**60. shuffle_rows**
```python
ops.shuffle_rows(input_tensor, dst2src_map)
```

#### MoE 算子

**61. moe_align_block_size**
```python
ops.moe_align_block_size(topk_ids, num_experts, block_size, sorted_token_ids, experts_ids, num_tokens_post_pad, expert_map=None)
```

**62. moe_sum**
```python
ops.moe_sum(input, output)
```

**63. topk_softmax**
```python
ops.topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize=False)
```

**64. grouped_topk**
```python
ops.grouped_topk(scores, num_expert_group, topk_group, topk, renormalize, routed_scaling_factor, bias, scoring_func=0)
```

**65. cutlass_moe_mm**
```python
ops.cutlass_moe_mm(out_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets, problem_sizes, a_strides, b_strides, c_strides, per_act_token, per_out_ch)
```

**66. cutlass_fp4_moe_mm**
```python
ops.cutlass_fp4_moe_mm(out_tensors, a_tensors, b_tensors, a_scales, b_scales, alphas, problem_sizes, expert_offsets, sf_offsets)
```

**67. get_cutlass_moe_mm_data**
```python
ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, problem_sizes1, problem_sizes2, input_permutation, output_permutation, num_experts, n, k, blockscale_offsets=None)
```

**68. get_cutlass_moe_mm_problem_sizes**
```python
ops.get_cutlass_moe_mm_problem_sizes(topk_ids, problem_sizes1, problem_sizes2, num_experts, n, k, blockscale_offsets=None, force_swap_ab=None)
```

**69. get_cutlass_pplx_moe_mm_data**
```python
ops.get_cutlass_pplx_moe_mm_data(expert_offsets, problem_sizes1, problem_sizes2, expert_num_tokens, num_local_experts, padded_m, n, k)
```

**70. moe_wna16_gemm**
```python
ops.moe_wna16_gemm(input, output, b_qweight, b_scales, b_qzeros, topk_weights, sorted_token_ids, experts_ids, num_tokens_post_pad, top_k, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, bit)
```

**71. moe_wna16_marlin_gemm**
```python
ops.moe_wna16_marlin_gemm(input, output, b_qweight, b_bias, b_scales, a_scales, global_scale, b_qzeros, g_idx, perm, workspace, sorted_token_ids, expert_ids, num_tokens_past_padded, topk_weights, moe_block_size, top_k, mul_topk_weights, is_ep, b_q_type, size_m, size_n, size_k, is_k_full, use_atomic_add, use_fp32_reduce, is_zp_float, thread_k=-1, thread_n=-1, blocks_per_sm=-1)
```

#### GGML 算子

**72. ggml_dequantize**
```python
ops.ggml_dequantize(W, quant_type, m, n, dtype)
```

**73. ggml_mul_mat_vec_a8**
```python
ops.ggml_mul_mat_vec_a8(W, X, quant_type, row)
```

**74. ggml_mul_mat_a8**
```python
ops.ggml_mul_mat_a8(W, X, quant_type, row)
```

**75. ggml_moe_a8**
```python
ops.ggml_moe_a8(X, W, sorted_token_ids, expert_ids, num_tokens_post_padded, quant_type, row, top_k, tokens)
```

**76. ggml_moe_a8_vec**
```python
ops.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row, tokens)
```

**77. ggml_moe_get_block_size**
```python
ops.ggml_moe_get_block_size(quant_type)
```

#### SSM / Mamba 算子

**78. selective_scan_fwd**
```python
ops.selective_scan_fwd(u, delta, A, B, C, D_, z_, delta_bias_, delta_softplus, query_start_loc, cache_indices, has_initial_state, ssm_states, pad_slot_id, block_size=1024, block_idx_first_scheduled_token=None, block_idx_last_scheduled_token=None, initial_state_idx=None)
```

#### Sampling / Repetition Penalties 算子

**79. apply_repetition_penalties**
```python
ops.apply_repetition_penalties(logits, prompt_mask, output_mask, repetition_penalties)
```

**80. apply_repetition_penalties_cuda**
```python
ops.apply_repetition_penalties_cuda(logits, prompt_mask, output_mask, repetition_penalties)
```

**81. apply_repetition_penalties_torch**
```python
ops.apply_repetition_penalties_torch(logits, prompt_mask, output_mask, repetition_penalties)
```

#### Device / Capability Query 算子

**82. get_device_attribute**
```python
ops.get_device_attribute(attribute, device)
```

**83. get_max_shared_memory_per_block_device_attribute**
```python
ops.get_max_shared_memory_per_block_device_attribute(device)
```

**84. cutlass_scaled_mm_supports_fp8**
```python
ops.cutlass_scaled_mm_supports_fp8(cuda_device_capability)
```

**85. cutlass_scaled_mm_supports_fp4**
```python
ops.cutlass_scaled_mm_supports_fp4(cuda_device_capability)
```

**86. cutlass_scaled_mm_supports_block_fp8**
```python
ops.cutlass_scaled_mm_supports_block_fp8(cuda_device_capability)
```

**87. cutlass_sparse_scaled_mm_supported**
```python
ops.cutlass_sparse_scaled_mm_supported(cuda_device_capability)
```

**88. cutlass_group_gemm_supported**
```python
ops.cutlass_group_gemm_supported(cuda_device_capability)
```

#### Communication / AllReduce 算子

**89. all_reduce**
```python
ops.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)
```

**90. init_custom_ar**
```python
ops.init_custom_ar(ipc_tensors, rank_data, rank, fully_connected)
```

**91. dispose**
```python
ops.dispose(fa)
```

**92. meta_size**
```python
ops.meta_size()
```

**93. register_buffer**
```python
ops.register_buffer(fa, ipc_tensors)
```

**94. get_graph_buffer_ipc_meta**
```python
ops.get_graph_buffer_ipc_meta(fa)
```

**95. register_graph_buffers**
```python
ops.register_graph_buffers(fa, handles, offsets)
```

**96. allocate_shared_buffer_and_handle**
```python
ops.allocate_shared_buffer_and_handle(size)
```

**97. open_mem_handle**
```python
ops.open_mem_handle(mem_handle)
```

**98. free_shared_buffer**
```python
ops.free_shared_buffer(ptr)
```

**99. qr_all_reduce**
```python
ops.qr_all_reduce(fa, inp, out, quant_level, cast_bf2half=False)
```

**100. qr_get_handle**
```python
ops.qr_get_handle(fa)
```

**101. qr_open_handles**
```python
ops.qr_open_handles(fa, handles)
```

**102. qr_max_size**
```python
ops.qr_max_size()
```

**103. qr_destroy**
```python
ops.qr_destroy(fa)
```

#### Other _custom_ops 算子

**104. hadacore_transform**
```python
ops.hadacore_transform(x, inplace=True)
```

### B类：直接通过 torch.ops._C 调用（25个）

> 无 `_custom_ops.py` 封装，在各模块中直接调用 `torch.ops._C.xxx(...)`

#### Activation 算子

**111. silu_and_mul**
```python
torch.ops._C.silu_and_mul(out, x)
# 调用位置: vllm/model_executor/layers/activation.py
```

**112. gelu_and_mul**
```python
torch.ops._C.gelu_and_mul(out, x)
# 调用位置: vllm/model_executor/layers/activation.py
```

**113. gelu_tanh_and_mul**
```python
torch.ops._C.gelu_tanh_and_mul(out, x)
# 调用位置: vllm/model_executor/layers/activation.py
```

**114. mul_and_silu**
```python
torch.ops._C.mul_and_silu(out, x)
# 调用位置: vllm/model_executor/layers/activation.py
```

**115. fatrelu_and_mul**
```python
torch.ops._C.fatrelu_and_mul(out, x, threshold)
# 调用位置: vllm/model_executor/layers/activation.py
```

**116. swigluoai_and_mul**
```python
torch.ops._C.swigluoai_and_mul(out, x, alpha, limit)
# 调用位置: vllm/model_executor/layers/activation.py
```

**117. gelu_new**
```python
torch.ops._C.gelu_new(out, x)
# 调用位置: vllm/model_executor/layers/activation.py
```

**118. gelu_fast**
```python
torch.ops._C.gelu_fast(out, x)
# 调用位置: vllm/model_executor/layers/activation.py
```

**119. gelu_quick**
```python
torch.ops._C.gelu_quick(out, x)
# 调用位置: vllm/model_executor/layers/activation.py
```

#### Fused Activation + Quantization 算子

**120. silu_and_mul_quant**
```python
torch.ops._C.silu_and_mul_quant(out, input, scale)
# 调用位置: vllm/compilation/activation_quant_fusion.py
```

**121. silu_and_mul_nvfp4_quant**
```python
torch.ops._C.silu_and_mul_nvfp4_quant(out, input, block_scale, global_scale)
# 调用位置: vllm/compilation/activation_quant_fusion.py
```

#### Fused Norm + Quantization 算子

**122. fused_add_rms_norm_static_fp8_quant**
```python
torch.ops._C.fused_add_rms_norm_static_fp8_quant(result, input, residual, weight, scale, epsilon)
# 调用位置: vllm/compilation/fusion.py
```

**123. rms_norm_static_fp8_quant**
```python
torch.ops._C.rms_norm_static_fp8_quant(result, input, weight, scale, epsilon)
# 调用位置: vllm/compilation/fusion.py
```

#### Low-level Quantization 算子

**124. dynamic_per_token_scaled_fp8_quant**
```python
torch.ops._C.dynamic_per_token_scaled_fp8_quant(output, input, scale, scale_ub)
# 调用位置: vllm/_custom_ops.py (scaled_fp8_quant内部)
```

**125. dynamic_scaled_fp8_quant**
```python
torch.ops._C.dynamic_scaled_fp8_quant(output, input, scale)
# 调用位置: vllm/_custom_ops.py (scaled_fp8_quant内部)
```

**126. static_scaled_fp8_quant**
```python
torch.ops._C.static_scaled_fp8_quant(output, input, scale)
# 调用位置: vllm/_custom_ops.py (scaled_fp8_quant内部，当scale.numel()==1时)
```

**127. static_scaled_int8_quant**
```python
torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)
# 调用位置: vllm/_custom_ops.py (scaled_int8_quant内部)
# azp: 非对称零点，对称量化时为None
```

**128. dynamic_scaled_int8_quant**
```python
torch.ops._C.dynamic_scaled_int8_quant(output, input.contiguous(), input_scales, input_azp)
# 调用位置: vllm/_custom_ops.py (scaled_int8_quant内部)
# input_scales: shape [num_tokens, 1], dtype=float32
# input_azp: dtype=int32 或 None（对称量化）
```

#### Other Low-level 算子

**129. per_token_group_quant_int8**
```python
torch.ops._C.per_token_group_quant_int8(x, x_q, x_s, group_size, eps, float(int8_min), float(int8_max))
# 调用位置: vllm/model_executor/layers/quantization/utils/int8_utils.py
# x_q: 输出量化tensor; x_s: 输出scale, shape [M, 1], dtype=float32
```

**130. top_k_per_row_prefill**
```python
torch.ops._C.top_k_per_row_prefill(logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices, num_rows, logits.stride(0), logits.stride(1), topk_tokens)
# 调用位置: vllm/model_executor/models/deepseek_v2.py
```

**131. top_k_per_row_decode**
```python
torch.ops._C.top_k_per_row_decode(logits, next_n, seq_lens, topk_indices, num_rows, logits.stride(0), logits.stride(1), topk_tokens)
# 调用位置: vllm/model_executor/models/deepseek_v2.py
```

**132. apply_repetition_penalties_**
```python
torch.ops._C.apply_repetition_penalties_(logits, prompt_mask, output_mask, repetition_penalties)
# 调用位置: vllm/_custom_ops.py (apply_repetition_penalties_cuda内部)
# 原地操作，直接修改logits
```

**133. persistent_masked_m_silu_mul_quant**
```python
torch.ops._C.persistent_masked_m_silu_mul_quant(y, tokens_per_expert, y_q, y_s, ceil_ue8m0)
# 调用位置: vllm/model_executor/layers/fused_moe/batched_deep_gemm_moe.py
# 融合SiLU激活+乘法+FP8量化，用于MoE，要求CUDA arch >= 80
```

**134. cutlass_fp4_group_mm**
```python
torch.ops._C.cutlass_fp4_group_mm(output, a, b, a_blockscale, b_blockscales, alphas, problem_sizes, expert_offsets, sf_offsets)
# 调用位置: vllm/_custom_ops.py (cutlass_fp4_moe_mm内部)
# 底层C++ kernel，cutlass_fp4_moe_mm(#66)的实际调用目标
```

**135. vllm_topk_softmax**
```python
torch.ops._moe_C.topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize)
# 调用位置: vllm/_custom_ops.py (topk_softmax函数内部)
# 注意: 使用 _moe_C 命名空间而非 _C
# 与A类#63 topk_softmax是同一底层kernel，此为torch.ops注册名
```

---

## 第三部分：不可运行算子（4个）

> 以下算子在 vllm v0.13 源码中有定义，但在当前环境中无法通过 API 访问。
> 原因包括：C++ extension 未正确注册、需要特定 GPU 架构编译、或依赖特定编译选项。

**1. weak_ref_tensor**
```python
# 源码位置: vllm/utils/torch_utils.py
# 调用方式: torch.ops._C.weak_ref_tensor(tensor)
# 注册位置: csrc/torch_bindings.cpp, dispatch key: CUDA
# 失败原因: torch.ops._C 中未注册
```

**2. get_cuda_view_from_cpu_tensor**
```python
# 源码位置: vllm/utils/torch_utils.py
# 调用方式: torch.ops._C.get_cuda_view_from_cpu_tensor(cpu_tensor)
# 注册位置: csrc/torch_bindings.cpp, dispatch key: CPU
# 失败原因: torch.ops._C 中未注册
```

**3. cutlass_mla_decode**
```python
# 源码位置: csrc/ops.h
# 调用方式: torch.ops._C.cutlass_mla_decode(out, q_nope, q_pe, kv_c_and_k_pe_cache, seq_lens, page_table, scale)
# 非SM100版本的CUTLASS MLA decode，与sm100_cutlass_mla_decode不同
# 失败原因: 可能需要特定GPU架构才会编译注册
```

**4. init_custom_qr**
```python
# 源码位置: csrc/custom_all_reduce.cu
# 调用方式: torch.ops._C_custom_ar.init_custom_qr(rank, world_size, qr_max_size)
# 初始化quick all reduce通信，使用 _C_custom_ar 命名空间
# 失败原因: _C_custom_ar 命名空间中未注册
```
