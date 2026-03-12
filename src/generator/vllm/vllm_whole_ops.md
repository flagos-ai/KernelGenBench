invoke_fused_moe_kernel	(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, A_scale: Optional[torch.Tensor], B_scale: Optional[torch.Tensor], B_zp: Optional[torch.Tensor], topk_weights: Optional[torch.Tensor], sorted_token_ids: torch.Tensor, expert_ids: torch.Tensor, num_tokens_post_padded: torch.Tensor, mul_routed_weight: bool, top_k: int, config: dict[str, typing.Any], compute_type: triton.language.core.dtype, use_fp8_w8a8: bool, use_int8_w8a8: bool, use_int8_w8a16: bool, use_int4_w4a16: bool, per_channel_quant: bool, block_shape: Optional[list[int]] = None, B_bias: Optional[torch.Tensor] = None) -> None	12267685
invoke_moe_batched_triton_kernel	(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, expert_num_tokens: torch.Tensor, compute_type: triton.language.core.dtype, A_scale: Optional[torch.Tensor], B_scale: Optional[torch.Tensor], B_zp: torch.Tensor, use_fp8_w8a8: bool, use_int8_w8a16: bool, use_int4_w4a16: bool, config: dict[str, int], per_act_token_quant: bool, block_shape: Optional[list[int]] = None)	12267685
triton_kernel_moe_forward	(hidden_states: torch.Tensor, w1, w2, gating_output: torch.Tensor, topk: int, renormalize: bool, activation: str = 'silu', quant_config: Optional[vllm.model_executor.layers.fused_moe.config.FusedMoEQuantConfig] = None, apply_router_weight_on_input: bool = False, global_num_experts: int = -1, expert_map: Optional[torch.Tensor] = None) -> torch.Tensor	12267685
triton_kernel_fused_experts	(output_tensor: torch.Tensor, hidden_states: torch.Tensor, w1, w2, routing_data, gather_indx, scatter_indx, activation: str = 'silu', quant_config: Optional[vllm.model_executor.layers.fused_moe.config.FusedMoEQuantConfig] = None, swiglu_alpha: float = 1.702, swiglu_limit: float = 7.0, apply_router_weight_on_input: bool = False, global_num_experts: int = -1, expert_map: Optional[torch.Tensor] = None, a1q_scale: Optional[torch.Tensor] = None) -> torch.Tensor	12181101
choose_scaled_mm_linear_kernel	(config: vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel.ScaledMMLinearLayerConfig, compute_capability: Optional[int] = None) -> type[vllm.model_executor.layers.quantization.kernels.scaled_mm.ScaledMMLinearKernel.ScaledMMLinearKernel]	7501851
batched_moe_kernel_quantize_input	(A: torch.Tensor, A_scale: Optional[torch.Tensor], num_tokens: int, E: int, N: int, expert_num_tokens: torch.Tensor, qtype: Optional[torch.dtype], per_act_token_quant: bool, block_shape: Optional[list[int]] = None) -> tuple[torch.Tensor, typing.Optional[torch.Tensor]]	7460438
moe_kernel_quantize_input	(A: torch.Tensor, A_scale: Optional[torch.Tensor], quant_dtype: Union[NoneType, torch.dtype, str], per_act_token_quant: bool, block_shape: Optional[list[int]] = None, is_fp4_scale_swizzled: bool = True) -> tuple[torch.Tensor, typing.Optional[torch.Tensor]]	7460438
triton_reshape_and_cache_flash	(key: torch.Tensor, value: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, slot_mapping: torch.Tensor, kv_cache_dtype: str, k_scale: torch.Tensor, v_scale: torch.Tensor)	7148670
torch_vllm_inplace_fused_experts	(**kwargs) -> torch.Tensor	6389704
torch_vllm_outplace_fused_experts	(**kwargs) -> torch.Tensor	6389704
fused_recurrent_gated_delta_rule_fwd	(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, scale: float, initial_state: torch.Tensor, inplace_final_state: bool = True, cu_seqlens: Optional[torch.LongTensor] = None, ssm_state_indices: Optional[torch.Tensor] = None, num_accepted_tokens: Optional[torch.Tensor] = None, use_qk_l2norm_in_kernel: bool = False) -> tuple[torch.Tensor, torch.Tensor]	5331380
fused_grouped_topk	(hidden_states: torch.Tensor, gating_output: torch.Tensor, topk: int, renormalize: bool, e_score_correction_bias: torch.Tensor, num_expert_group: int = 0, topk_group: int = 0, scoring_func: str = 'softmax', routed_scaling_factor: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]	5139911
fused_topk	(hidden_states: torch.Tensor, gating_output: torch.Tensor, topk: int, renormalize: bool, indices_type: Optional[torch.dtype] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]	5131511
fused_topk_bias	(hidden_states: torch.Tensor, gating_output: torch.Tensor, e_score_correction_bias: torch.Tensor, topk: int, renormalize: bool)	5131511
modular_triton_fused_moe	(quant_config: vllm.model_executor.layers.fused_moe.config.FusedMoEQuantConfig) -> vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEModularKernel	5131511
fused_add_rms_norm_static_fp8_quant	torch::Tensor& out, torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, torch::Tensor& scale, double epsilon	5022194
fused_add_rms_norm	torch::Tensor& input, torch::Tensor& residual, torch::Tensor& weight, double epsilon	5017394
fused_qk_norm_rope	torch::Tensor& qkv, int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_dim, double eps, torch::Tensor& q_weight, torch::Tensor& k_weight, torch::Tensor& cos_sin_cache, bool is_neox, torch::Tensor& position_ids	5017390
awq_gemm_triton	(input: torch.Tensor, qweight: torch.Tensor, scales: torch.Tensor, qzeros: torch.Tensor, split_k_iters: int, block_size_m: int = 32, block_size_n: int = 32, block_size_k: int = 32) -> torch.Tensor	4977877
fused_recurrent_gated_delta_rule	(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor = None, scale: float = None, initial_state: torch.Tensor = None, inplace_final_state: bool = True, cu_seqlens: Optional[torch.LongTensor] = None, ssm_state_indices: Optional[torch.Tensor] = None, num_accepted_tokens: Optional[torch.Tensor] = None, use_qk_l2norm_in_kernel: bool = False) -> tuple[torch.Tensor, torch.Tensor]	4975952
triton_scaled_mm	(input: torch.Tensor, weight: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, out_dtype: type[torch.dtype], bias: Optional[torch.Tensor] = None, block_size_m: int = 32, block_size_n: int = 32, block_size_k: int = 32, use_heuristic=True) -> torch.Tensor	4946148
fused_experts_impl	(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, inplace: bool = False, activation: str = 'silu', apply_router_weight_on_input: bool = False, use_fp8_w8a8: bool = False, use_int8_w8a8: bool = False, use_int8_w8a16: bool = False, use_int4_w4a16: bool = False, use_mxfp4_w4a4: bool = False, per_channel_quant: bool = False, global_num_experts: int = -1, expert_map: Optional[torch.Tensor] = None, w1_scale: Optional[torch.Tensor] = None, w2_scale: Optional[torch.Tensor] = None, w1_zp: Optional[torch.Tensor] = None, w2_zp: Optional[torch.Tensor] = None, a1_scale: Optional[torch.Tensor] = None, a2_scale: Optional[torch.Tensor] = None, block_shape: Optional[list[int]] = None, w1_bias: Optional[torch.Tensor] = None, w2_bias: Optional[torch.Tensor] = None) -> torch.Tensor	4927021
fused_gdn_gating	(A_log: torch.Tensor, a: torch.Tensor, dt_bias: torch.Tensor, beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor	4926929
dispatch_fused_experts_func	(inplace: bool) -> Callable[..., torch.Tensor]	4926821
fused_experts	(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, inplace: bool = False, activation: str = 'silu', apply_router_weight_on_input: bool = False, global_num_experts: int = -1, expert_map: Optional[torch.Tensor] = None, quant_config: Optional[vllm.model_executor.layers.fused_moe.config.FusedMoEQuantConfig] = None, allow_deep_gemm: bool = False, allow_cutlass_block_scaled_grouped_gemm: bool = False) -> torch.Tensor	4926821
inplace_fused_experts	(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, activation: str = 'silu', apply_router_weight_on_input: bool = False, use_fp8_w8a8: bool = False, use_int8_w8a8: bool = False, use_int8_w8a16: bool = False, use_int4_w4a16: bool = False, use_mxfp4_w4a4: bool = False, per_channel_quant: bool = False, global_num_experts: int = -1, expert_map: Optional[torch.Tensor] = None, w1_scale: Optional[torch.Tensor] = None, w2_scale: Optional[torch.Tensor] = None, w1_zp: Optional[torch.Tensor] = None, w2_zp: Optional[torch.Tensor] = None, a1_scale: Optional[torch.Tensor] = None, a2_scale: Optional[torch.Tensor] = None, block_shape: Optional[List[int]] = None, w1_bias: Optional[torch.Tensor] = None, w2_bias: Optional[torch.Tensor] = None) -> None	4926821
inplace_fused_experts_fake	(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, activation: str = 'silu', apply_router_weight_on_input: bool = False, use_fp8_w8a8: bool = False, use_int8_w8a8: bool = False, use_int8_w8a16: bool = False, use_int4_w4a16: bool = False, use_mxfp4_w4a4: bool = False, per_channel_quant: bool = False, global_num_experts: int = -1, expert_map: Optional[torch.Tensor] = None, w1_scale: Optional[torch.Tensor] = None, w2_scale: Optional[torch.Tensor] = None, w1_zp: Optional[torch.Tensor] = None, w2_zp: Optional[torch.Tensor] = None, a1_scale: Optional[torch.Tensor] = None, a2_scale: Optional[torch.Tensor] = None, block_shape: Optional[List[int]] = None, w1_bias: Optional[torch.Tensor] = None, w2_bias: Optional[torch.Tensor] = None) -> None	4926821
outplace_fused_experts	(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, activation: str = 'silu', apply_router_weight_on_input: bool = False, use_fp8_w8a8: bool = False, use_int8_w8a8: bool = False, use_int8_w8a16: bool = False, use_int4_w4a16: bool = False, use_mxfp4_w4a4: bool = False, per_channel_quant: bool = False, global_num_experts: int = -1, expert_map: Optional[torch.Tensor] = None, w1_scale: Optional[torch.Tensor] = None, w2_scale: Optional[torch.Tensor] = None, w1_zp: Optional[torch.Tensor] = None, w2_zp: Optional[torch.Tensor] = None, a1_scale: Optional[torch.Tensor] = None, a2_scale: Optional[torch.Tensor] = None, block_shape: Optional[List[int]] = None, w1_bias: Optional[torch.Tensor] = None, w2_bias: Optional[torch.Tensor] = None) -> torch.Tensor	4926821
outplace_fused_experts_fake	(hidden_states: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, activation: str = 'silu', use_fp8_w8a8: bool = False, use_int8_w8a8: bool = False, use_int8_w8a16: bool = False, use_int4_w4a16: bool = False, use_mxfp4_w4a4: bool = False, per_channel_quant: bool = False, global_num_experts: int = -1, expert_map: Optional[torch.Tensor] = None, w1_scale: Optional[torch.Tensor] = None, w2_scale: Optional[torch.Tensor] = None, w1_zp: Optional[torch.Tensor] = None, w2_zp: Optional[torch.Tensor] = None, a1_scale: Optional[torch.Tensor] = None, a2_scale: Optional[torch.Tensor] = None, block_shape: Optional[list[int]] = None, w1_bias: Optional[torch.Tensor] = None, w2_bias: Optional[torch.Tensor] = None) -> torch.Tensor	4926821
triton_convert_req_index_to_global_index	(req_id: torch.Tensor, block_table: torch.Tensor, token_indices: torch.Tensor, BLOCK_SIZE: int = 64, NUM_TOPK_TOKENS: int = 2048, BLOCK_N: int = 128)	4902820
zero_experts_compute_triton	(expert_indices: torch.Tensor, expert_scales: torch.Tensor, num_experts: int, zero_expert_type: str, hidden_states: torch.Tensor) -> torch.Tensor	4818357
linear_decode_forward_triton	(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_caches: torch.Tensor, slope_rate: torch.Tensor, slot_idx: torch.Tensor, BLOCK_SIZE: int = 32) -> torch.Tensor	4815957
awq_dequantize_triton	(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, block_size_x: int = 32, block_size_y: int = 32) -> torch.Tensor	4811757
pack_seq_triton	(x: torch.Tensor, lengths: torch.Tensor, pad_value: float = -inf, block_t: int = 64, block_d: int = 64) -> torch.Tensor	4811757
triton_mrope	(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mrope_section: list[int], head_size: int, rotary_dim: int, mrope_interleaved: bool) -> tuple[torch.Tensor, torch.Tensor]	4811757
unpack_seq_triton	(packed_tensor: torch.Tensor, lengths: torch.Tensor, block_t: int = 64, block_d: int = 64) -> torch.Tensor	4811757
silu_and_mul_scaled_fp4_experts_quant	torch::Tensor& output, torch::Tensor& output_scale, torch::Tensor const& input, torch::Tensor const& input_global_scale, torch::Tensor const& input_offset_by_experts, torch::Tensor const& output_scale_offset_by_experts	4455797
silu_and_mul_nvfp4_quant	torch::Tensor& out, torch::Tensor& output_block_scale, torch::Tensor& input, torch::Tensor& input_global_scale	4326206
silu_and_mul_quant	torch::Tensor& out, torch::Tensor& input, torch::Tensor& scale	4326206
gelu_and_mul	torch::Tensor& out, torch::Tensor& input	4321406
gelu_tanh_and_mul	torch::Tensor& out, torch::Tensor& input	4321406
mul_and_silu	torch::Tensor& out, torch::Tensor& input	4321406
silu_and_mul	torch::Tensor& out, torch::Tensor& input	4321406
fatrelu_and_mul	torch::Tensor& out, torch::Tensor& input, double threshold	4304804
swigluoai_and_mul	torch::Tensor& out, torch::Tensor& input, double alpha = 1.702, double limit = 7.0	4304804
cutlass_encode_and_reorder_int4b		3481155
cutlass_encode_and_reorder_int4b_fake		3481155
silu_mul_fp8_quant_deep_gemm_cuda	(y: torch.Tensor, tokens_per_expert: torch.Tensor, num_parallel_tokens=16, group_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]	3424179
persistent_masked_m_silu_mul_quant	const at::Tensor& input, // (E, T, 2*H) const at::Tensor& counts, // (E) at::Tensor& y_q, // (E, T, H) [OUT] at::Tensor& y_s, // (E, T, H//group_size) [OUT] bool use_ue8m0	3259867
activation_without_mul	(activation: str) -> str	3206655
ggml_mul_mat_a8	torch::Tensor W, torch::Tensor X, int64_t type, int64_t row	3206655
ggml_mul_mat_vec_a8	torch::Tensor W, torch::Tensor X, int64_t type, int64_t row	3206655
get_cutlass_moe_mm_problem_sizes		2888010
get_cutlass_moe_mm_problem_sizes_from_expert_offsets	const torch::Tensor& expert_first_token_offset, torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2, const int64_t n, const int64_t k, const bool swap_ab	2888010
get_max_shared_memory_per_block_device_attribute		2735887
get_cutlass_moe_mm_data	const torch::Tensor& topk_ids, torch::Tensor& expert_offsets, torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2, torch::Tensor& input_permutation, torch::Tensor& output_permutation, const int64_t num_experts, const int64_t n, const int64_t k, const std::optional<torch::Tensor>& blockscale_offsets	2721190
get_cutlass_pplx_moe_mm_data	torch::Tensor& expert_offsets, torch::Tensor& problem_sizes1, torch::Tensor& problem_sizes2, const torch::Tensor& expert_num_tokens, const int64_t num_local_experts, const int64_t padded_m, const int64_t n, const int64_t k	2721190
cutlass_fp4_moe_mm		2689615
cutlass_moe_mm	torch::Tensor& out_tensors, torch::Tensor const& a_tensors, torch::Tensor const& b_tensors, torch::Tensor const& a_scales, torch::Tensor const& b_scales, torch::Tensor const& expert_offsets, torch::Tensor const& problem_sizes, torch::Tensor const& a_strides, torch::Tensor const& b_strides, torch::Tensor const& c_strides, bool per_act_token, bool per_out_ch	2684815
decode_attention_fwd_grouped	(q, k_buffer, v_buffer, o, lse, req_to_token, b_seq_len, attn_logits, num_kv_splits, sm_scale, page_size, logit_cap=0.0)	2676680
decode_attention_fwd	(q, k_buffer, v_buffer, o, lse, req_to_token, b_seq_len, attn_logits, num_kv_splits, sm_scale, page_size=1, logit_cap=0.0)	2668280
decode_attention_fwd_normal	(q, k_buffer, v_buffer, o, lse, req_to_token, b_seq_len, attn_logits, num_kv_splits, sm_scale, page_size, logit_cap=0.0)	2668280
cutlass_scaled_mm_supports_block_fp8	int64_t cuda_device_capability	2560904
cutlass_blockwise_scaled_grouped_mm		2510506
cutlass_scaled_fp4_mm	torch::Tensor& D, torch::Tensor const& A, torch::Tensor const& B, torch::Tensor const& A_sf, torch::Tensor const& B_sf, torch::Tensor const& alpha	2502106
cutlass_scaled_mm	torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b, torch::Tensor const& a_scales, torch::Tensor const& b_scales, std::optional<torch::Tensor> const& bias	2502106
cutlass_scaled_mm_azp	torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b, torch::Tensor const& a_scales, torch::Tensor const& b_scales, torch::Tensor const& azp_adj, std::optional<torch::Tensor> const& azp, std::optional<torch::Tensor> const& bias	2502106
cutlass_scaled_mm_supports_fp4	int64_t cuda_device_capability	2502106
cutlass_scaled_mm_supports_fp8	int64_t cuda_device_capability	2502106
cutlass_scaled_sparse_mm	torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b, torch::Tensor const& e, torch::Tensor const& a_scales, torch::Tensor const& b_scales, std::optional<torch::Tensor> const& bias	2502106
cutlass_sparse_scaled_mm_supported	int64_t cuda_device_capability	2502106
flash_mla_with_kvcache		2428533
sm100_cutlass_mla_get_workspace_size		2417657
cutlass_fp4_group_mm	torch::Tensor& output, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& a_blockscale, const torch::Tensor& b_blockscales, const torch::Tensor& alphas, const torch::Tensor& problem_sizes, const torch::Tensor& expert_offsets, const torch::Tensor& sf_offsets	2377825
cutlass_w4a8_mm		2367715
cutlass_w4a8_mm_fake		2367715
cutlass_group_gemm_supported	int64_t cuda_device_capability	2339778
reshape_and_cache_flash		2336948
cutlass_pack_scale_fp8		2326608
cutlass_pack_scale_fp8_fake		2326608
cutlass_mla_decode	torch::Tensor const& out, torch::Tensor const& q_nope, torch::Tensor const& q_pe, torch::Tensor const& kv_c_and_k_pe_cache, torch::Tensor const& seq_lens, torch::Tensor const& page_table, double scale	2322384
sm100_cutlass_mla_decode		2322384
cutlass_sparse_compress	torch::Tensor const& a	2321808
get_flash_mla_metadata		2297128
get_device_attribute		2271439
gdn_attention	(hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str) -> None	2263829
gdn_attention_fake	(hidden_states: torch.Tensor, output: torch.Tensor, layer_name: str) -> None	2263829
paged_attention_rocm		2260753
paged_attention_v1	torch::Tensor& out, torch::Tensor& query, torch::Tensor& key_cache, torch::Tensor& value_cache, int64_t num_kv_heads, double scale, torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size, int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes, const std::string& kv_cache_dtype, torch::Tensor& k_scale, torch::Tensor& v_scale, const int64_t tp_rank, const int64_t blocksparse_local_blocks, const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size, const int64_t blocksparse_head_sliding_step	2260753
paged_attention_v2	torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits, torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache, torch::Tensor& value_cache, int64_t num_kv_heads, double scale, torch::Tensor& block_tables, torch::Tensor& seq_lens, int64_t block_size, int64_t max_seq_len, const std::optional<torch::Tensor>& alibi_slopes, const std::string& kv_cache_dtype, torch::Tensor& k_scale, torch::Tensor& v_scale, const int64_t tp_rank, const int64_t blocksparse_local_blocks, const int64_t blocksparse_vert_stride, const int64_t blocksparse_block_size, const int64_t blocksparse_head_sliding_step	2260753
lightning_attention	(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ed: torch.Tensor, block_size: int = 256, kv_history: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]	2260193
unified_attention	(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, softmax_scale, causal, window_size, block_table, softcap, q_descale, k_descale, v_descale, alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None)	2260193
block_quant_to_tensor_quant	(x_q_block: torch.Tensor, x_s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]	2187225
expand_batch_to_tokens	(x: torch.Tensor, cu_num_tokens: torch.Tensor, num_tokens: int, replace_from: int = 0, replace_to: int = 0) -> torch.Tensor	2182941
copy_blocks_mla		2171145
copy_blocks		2170569
input_to_int8	(x: torch.Tensor, dtype: torch.dtype = torch.int8) -> tuple[torch.Tensor, torch.Tensor]	2128399
_lse2_to_lse	(lse_base2: torch.Tensor) -> torch.Tensor	2123599
input_to_float8	(x: torch.Tensor, dtype: Optional[torch.dtype] = None) -> tuple[torch.Tensor, torch.Tensor]	2123599
is_int_pow_2	(n)	2120864
vllm_topk_softmax	(topk_weights: torch.Tensor, topk_indices: torch.Tensor, token_expert_indices: torch.Tensor, gating_output: torch.Tensor, renormalize: bool) -> tuple[torch.Tensor, ...]	1600789
expert_num_tokens_round_up_and_sum	(expert_num_tokens: torch.Tensor, alignment: int) -> int	1503776
deepgemm_unpermute_and_reduce	(a: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, inv_perm: torch.Tensor, expert_map: Optional[torch.Tensor], output: torch.Tensor)	1233029
gather_and_maybe_dequant_cache		1189837
indexer_k_quant_and_cache		1164147
allocate_shared_buffer_and_handle		1163857
concat_and_cache_mla		1159363
reshape_and_cache		1159347
cp_gather_cache		1113642
tensor_cache	(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]	1083180
_resize_cache	(x: torch.Tensor, v: tuple[int, ...]) -> torch.Tensor	1083152
chunked_prefill_paged_decode	(query, key, value, output, kv_cache_dtype, key_cache, value_cache, block_table, query_start_loc, seq_lens, max_seq_len, max_query_len, k_scale, v_scale, alibi_slopes=None, sliding_window=None, sm_scale=None, output_scale=None, sinks=None)	1083152
gelu_new	torch::Tensor& out, torch::Tensor& input	986552
gelu_fast	torch::Tensor& out, torch::Tensor& input	956264
gelu_quick	torch::Tensor& out, torch::Tensor& input	956264
chunk_scaled_dot_kkt_fwd	(k: torch.Tensor, beta: torch.Tensor, g_cumsum: Optional[torch.Tensor] = None, cu_seqlens: Optional[torch.LongTensor] = None, chunk_size: int = 64, output_dtype: torch.dtype = torch.float32) -> torch.Tensor	526075
marlin_gemm_moe_fake		514596
moe_wna16_marlin_gemm		514596
moe_wna16_marlin_gemm_fake		514596
_layer_norm_fwd	(x, weight, bias, eps, z=None, out=None, group_size=None, norm_before_gate=True, is_rms_norm=False)	513962
layer_norm_fwd	(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, z: torch.Tensor = None, out: torch.Tensor = None, group_size: int = None, norm_before_gate: bool = True, is_rms_norm: bool = False)	513962
moe_wna16_gemm		495880
_mamba_chunk_scan_combined_fwd	(x, dt, A, B, C, chunk_size, out, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, chunk_indices=None, chunk_offsets=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, inf), state_dtype=None)	436659
_bmm_chunk_fwd	(a, b, chunk_size, seq_idx, causal=False, output_dtype=None)	428365
_chunk_cumsum_fwd	(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, inf))	428259
_chunk_scan_fwd	(cb, x, dt, dA_cumsum, C, states, out, seq_idx, D=None, z=None, chunk_indices=None, chunk_offsets=None, initial_states=None)	428259
_chunk_state_fwd	(B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True)	428259
chunk_fwd_o	(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, h: torch.Tensor, g: Optional[torch.Tensor] = None, scale: Optional[float] = None, cu_seqlens: Optional[torch.LongTensor] = None, chunk_size: int = 64) -> torch.Tensor	428259
chunk_gated_delta_rule_fwd_h	(k: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: Optional[torch.Tensor] = None, initial_state: Optional[torch.Tensor] = None, output_final_state: bool = False, chunk_size: int = 64, save_new_value: bool = True, cu_seqlens: Optional[torch.LongTensor] = None) -> tuple[torch.Tensor, torch.Tensor]	428259
selective_scan_fwd	const torch::Tensor& u, const torch::Tensor& delta, const torch::Tensor& A, const torch::Tensor& B, const torch::Tensor& C, const std::optional<torch::Tensor>& D_, const std::optional<torch::Tensor>& z_, const std::optional<torch::Tensor>& delta_bias_, bool delta_softplus, const std::optional<torch::Tensor>& query_start_loc, const std::optional<torch::Tensor>& cache_indices, const std::optional<torch::Tensor>& has_initial_state, const torch::Tensor& ssm_states, int64_t pad_slot_id, int64_t block_size, const std::optional<torch::Tensor>& block_idx_first_scheduled_token, const std::optional<torch::Tensor>& block_idx_last_scheduled_token, const std::optional<torch::Tensor>& initial_state_idx	428259
_decode_grouped_att_m_fwd	(q, k_buffer, v_buffer, att_out, Req_to_tokens, B_Seqlen, num_kv_splits, sm_scale, page_size, logit_cap)	416487
_decode_softmax_reducev_fwd	(logits, q, o, lse, v_buffer, b_seq_len, num_kv_splits)	411287
_decode_att_m_fwd	(q, k_buffer, v_buffer, att_out, Req_to_tokens, B_Seqlen, num_kv_splits, sm_scale, page_size, logit_cap)	408087
_state_passing_fwd	(states, dA_cumsum, seq_idx, chunk_offsets, initial_states=None, out_dtype=None)	408087
l2norm_fwd	(x: torch.Tensor, eps: float = 1e-06, output_dtype: Optional[torch.dtype] = None)	408087
recompute_w_u_fwd	(k: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, g_cumsum: torch.Tensor, A: torch.Tensor, cu_seqlens: Optional[torch.LongTensor]) -> tuple[torch.Tensor, torch.Tensor]	408087
prepare_chunk_indices	(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor	372352
deepgemm_moe_permute	(aq: torch.Tensor, aq_scale: torch.Tensor, topk_ids: torch.Tensor, local_num_experts: int, expert_map: Optional[torch.Tensor], expert_tokens_meta: Optional[vllm.model_executor.layers.fused_moe.modular_kernel.ExpertTokensMetadata], aq_out: Optional[torch.Tensor] = None)	368900
supports_moe_ops		368018
count_expert_num_tokens	(topk_ids: torch.Tensor, num_local_experts: int, expert_map: Optional[torch.Tensor]) -> torch.Tensor	353587
ggml_moe_get_block_size	int64_t type	353575
get_moe_wna16_block_config	(config: dict[str, int], use_moe_wna16_cuda: bool, num_valid_tokens: int, size_k: int, size_n: int, num_experts: int, group_size: int, real_top_k: int, block_size_m: int)	353475
try_get_optimal_moe_config	(w1_shape: tuple[int, ...], w2_shape: tuple[int, ...], top_k: int, dtype: Optional[str], M: int, block_shape: Optional[list[int]] = None) -> dict[str, int]	353475
mamba_chunk_scan_combined_varlen	(x, dt, A, B, C, chunk_size, cu_seqlens, seq_idx, out, D=None, z=None, dt_bias=None, initial_states=None, chunk_indices=None, chunk_offsets=None, dt_softplus=False, dt_limit=(0.0, inf), state_dtype=None)	349000
awq_marlin_moe_repack		348476
gptq_marlin_moe_repack		348476
should_moe_wna16_use_cuda	(num_valid_tokens: int, group_size: int, num_experts: int, bit: int)	347102
prepare_chunk_offsets	(cu_seqlens: torch.LongTensor, chunk_size: int) -> torch.LongTensor	342200
chunk_state_varlen	(B, x, dt, dA_cumsum, cu_seqlens, chunk_states, initial_states=None)	340600
grouped_topk		325500
moe_sum		322900
dynamic_4bit_int_moe_cpu	torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights, torch::Tensor w13_packed, torch::Tensor w2_packed, int64_t H, int64_t I, int64_t I2, int64_t group_size, bool apply_router_weight_on_input, int64_t activation_kind	321900
routing_from_bitmatrix	(bitmatrix, expt_scal, expt_indx, n_expts_tot, n_expts_act)	321300
topk_softmax		320300
moe_align_block_size		317200
dispatch_topk_func	() -> Callable[..., tuple[torch.Tensor, ...]]	317100
expert_weight_is_col_major	(x: torch.Tensor) -> bool	317100
ggml_moe_a8	torch::Tensor X, torch::Tensor W, torch::Tensor sorted_token_ids, torch::Tensor expert_ids, torch::Tensor num_tokens_post_padded, int64_t type, int64_t row, int64_t top_k, int64_t tokens	317100
ggml_moe_a8_vec	torch::Tensor X, torch::Tensor W, torch::Tensor topk_ids, int64_t top_k, int64_t type, int64_t row, int64_t tokens	317100
make_routing_data	(topk_ids: torch.Tensor, topk_weights: torch.Tensor, num_local_experts: int) -> tuple['RoutingData', torch.Tensor, torch.Tensor]	317100
routing	(logits, n_expts_act, sm_first=False, expt_indx=None, simulated_ep=1, n_rows=None)	317100
chunk_local_cumsum_scalar	(g: torch.Tensor, chunk_size: int, reverse: bool = False, cu_seqlens: Optional[torch.Tensor] = None, head_first: bool = False, output_dtype: Optional[torch.dtype] = torch.float32) -> torch.Tensor	316157
dynamic_per_token_scaled_fp8_quant	torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale, std::optional<torch::Tensor> const& scale_ub	305493
chunk_local_cumsum	(g: torch.Tensor, chunk_size: int, reverse: bool = False, cu_seqlens: Optional[torch.Tensor] = None, head_first: bool = False, output_dtype: Optional[torch.dtype] = torch.float32, **kwargs) -> torch.Tensor	304113
chunk_local_cumsum_vector	(g: torch.Tensor, chunk_size: int, reverse: bool = False, cu_seqlens: Optional[torch.Tensor] = None, head_first: bool = False, output_dtype: Optional[torch.dtype] = torch.float32) -> torch.Tensor	304113
selective_scan_fn	(u, ssm_states, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, query_start_loc=None, cache_indices=None, has_initial_state=None, pad_slot_id=-1) -> torch.Tensor	304113
get_cuda_view_from_cpu_tensor	torch::Tensor& cpu_tensor	271468
create_onednn_scaled_mm		266898
onednn_scaled_mm		266898
rms_norm_per_block_quant	torch::Tensor& out, torch::Tensor const& input, torch::Tensor const& weight, torch::Tensor& scales, double const epsilon, std::optional<torch::Tensor> scale_ub, std::optional<torch::Tensor> residual, int64_t group_size, bool is_scale_transposed	247628
dynamic_scaled_fp8_quant	torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scale	220991
dynamic_scaled_int8_quant	torch::Tensor& out, torch::Tensor const& input, torch::Tensor& scales, std::optional<torch::Tensor> const& azp	220991
onednn_scaled_int8_quant		220991
scaled_fp4_experts_quant	torch::Tensor& output, torch::Tensor& output_scale, torch::Tensor const& input, torch::Tensor const& input_global_scale, torch::Tensor const& input_offset_by_experts, torch::Tensor const& output_scale_offset_by_experts	220991
scaled_fp4_quant	torch::Tensor& output, torch::Tensor const& input, torch::Tensor& output_scale, torch::Tensor const& input_scale, bool is_sf_swizzled_layout	220991
scaled_fp8_quant		220991
scaled_int8_quant		220991
static_scaled_fp8_quant	torch::Tensor& out, torch::Tensor const& input, torch::Tensor const& scale, std::optional<std::tuple<int64_t, int64_t>> group_shape = std::nullopt	220991
static_scaled_int8_quant	torch::Tensor& out, torch::Tensor const& input, torch::Tensor const& scale, std::optional<torch::Tensor> const& azp	220991
gptq_marlin_24_gemm		207192
gptq_marlin_gemm		207192
machete_mm		195227
machete_mm_fake		195227
rms_norm_dynamic_per_token_quant	torch::Tensor& out, torch::Tensor const& input, torch::Tensor const& weight, torch::Tensor& scales, double const epsilon, std::optional<torch::Tensor> scale_ub, std::optional<torch::Tensor> residual	188830
rocm_aiter_gemm_w8a8_blockscale_impl	(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor, block_size: list[int], output_dtype: torch.dtype = torch.float16) -> torch.Tensor	178980
allspark_w8a16_gemm		178780
awq_gemm	torch::Tensor _in_feats, torch::Tensor _kernel, torch::Tensor _scaling_factors, torch::Tensor _zeros, int64_t split_k_iters	178780
gptq_gemm	torch::Tensor a, torch::Tensor b_q_weight, torch::Tensor b_gptq_qzeros, torch::Tensor b_gptq_scales, torch::Tensor b_g_idx, bool use_exllama, bool use_v2_format, int64_t bit	178780
rocm_aiter_gemm_w8a8_blockscale_fake	(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor, block_size: list[int], output_dtype: torch.dtype = torch.float16) -> torch.Tensor	178780
rms_norm_gated	(x, weight, bias, z=None, eps=1e-06, group_size=None, norm_before_gate=True)	158534
qr_max_size		147098
create_onednn_mm		132507
onednn_mm		132507
get_all_max_shared_mem	()	129085
rms_norm_static_fp8_quant	torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, torch::Tensor& scale, double epsilon	110675
poly_norm		105875
rms_norm	torch::Tensor& out, torch::Tensor& input, torch::Tensor& weight, double epsilon	105875
rms_norm_ref	(x, weight, bias, z=None, eps=1e-06, group_size=None, norm_before_gate=True, upcast=True)	105875
per_token_group_quant_8bit_packed	const torch::Tensor& input, torch::Tensor& output_q, torch::Tensor& output_s_packed, int64_t group_size, double eps, double min_8bit, double max_8bit	99412
per_token_group_quant_fp8	const torch::Tensor& input, torch::Tensor& output_q, torch::Tensor& output_s, int64_t group_size, double eps, double fp8_min, double fp8_max, bool scale_ue8m0	99412
per_token_group_quant_int8	const torch::Tensor& input, torch::Tensor& output_q, torch::Tensor& output_s, int64_t group_size, double eps, double int8_min, double int8_max	99412
per_token_quant_int8	(x)	89302
top_k_per_row_decode	const torch::Tensor& logits, int64_t next_n, const torch::Tensor& seqLens, torch::Tensor& indices, int64_t numRows, int64_t stride0, int64_t stride1, int64_t topK	89302
top_k_per_row_prefill	const torch::Tensor& logits, const torch::Tensor& rowStarts, const torch::Tensor& rowEnds, torch::Tensor& indices, int64_t numRows, int64_t stride0, int64_t stride1, int64_t topK	89302
machete_prepack_B		82127
machete_prepack_B_fake		82127
machete_supported_schedules		82127
all_reduce	fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, fptr_t reg_buffer, int64_t reg_buffer_sz_bytes	73682
qr_all_reduce	fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, int64_t quant_level, bool cast_bf2half = false	73682
w8a8_block_fp8_matmul	(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor, block_size: list[int], output_dtype: torch.dtype = torch.float16) -> torch.Tensor	71998
w8a8_block_int8_matmul	(A: torch.Tensor, B: torch.Tensor, As: torch.Tensor, Bs: torch.Tensor, block_size: list[int], output_dtype: torch.dtype = torch.float16) -> torch.Tensor	71998
apply_fp8_block_linear	(layer: torch.nn.modules.module.Module, input: torch.Tensor, bias: Optional[torch.Tensor], cutlass_block_fp8_supported: bool, use_aiter_and_is_supported: bool) -> torch.Tensor	63598
apply_w8a8_block_fp8_linear	(input: torch.Tensor, weight: torch.Tensor, block_size: list[int], weight_scale: torch.Tensor, input_scale: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None, cutlass_block_fp8_supported: bool = False, use_aiter_and_is_supported: bool = False) -> torch.Tensor	63598
apply_w8a8_block_fp8_linear_fake	(input: torch.Tensor, weight: torch.Tensor, block_size: list[int], weight_scale: torch.Tensor, input_scale: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None, cutlass_block_fp8_supported: bool = False, use_aiter_and_is_supported: bool = False) -> torch.Tensor	63598
apply_w8a8_block_int8_linear	(input: torch.Tensor, weight: torch.Tensor, block_size: list[int], weight_scale: torch.Tensor, input_scale: Optional[torch.Tensor] = None, bias: Optional[torch.Tensor] = None) -> torch.Tensor	63598
maybe_post_process_fp8_weight_block	(layer: torch.nn.modules.module.Module, cutlass_block_fp8_supported: bool)	63598
process_fp8_weight_block_strategy	(weight: torch.Tensor, weight_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]	63598
validate_fp8_block_shape	(layer: torch.nn.modules.module.Module, input_size: int, output_size: int, input_size_per_partition: int, output_partition_sizes: list[int], block_size: list[int]) -> None	63598
meta_size		58898
block_dequant	(x_q_block: torch.Tensor, x_s: torch.Tensor, block_size: list[int]) -> torch.Tensor	58798
merge_attn_states	torch::Tensor& output, std::optional<torch::Tensor> output_lse, const torch::Tensor& prefix_output, const torch::Tensor& prefix_lse, const torch::Tensor& suffix_output, const torch::Tensor& suffix_lse	52781
permute_cols	torch::Tensor const& A, torch::Tensor const& perm	51800
_get_config_quant_dtype	(use_fp8_w8a8: bool, use_int8_w8a8: bool, use_mxfp4_w4a4: bool) -> Union[NoneType, torch.dtype, str]	41175
awq_marlin_repack		41072
gptq_marlin_repack		41072
prepare_lens	(cu_seqlens: torch.LongTensor) -> torch.LongTensor	38087
swap_blocks		36487
get_graph_buffer_ipc_meta		36381
_get_trtllm_gen_workspace_buffer	()	36375
get_autotune_configs	()	36375
get_cdna_autotune_configs	()	36375
get_config_file_name	(E: int, N: int, dtype: Optional[str], block_shape: Optional[list[int]] = None) -> str	36375
get_default_config	(M: int, E: int, N: int, K: int, topk: int, dtype: Optional[str], block_shape: Optional[list[int]] = None) -> dict[str, int]	36375
get_rdna_autotune_configs	()	36375
qr_get_handle	fptr_t _fa	36375
apply_repetition_penalties_cuda		30002
rotary_embedding	torch::Tensor& positions, torch::Tensor& query, std::optional<torch::Tensor> key, int64_t head_size, torch::Tensor& cos_sin_cache, bool is_neox	26427
sample_recovered_tokens	(max_spec_len: int, num_draft_tokens: list[int], cu_num_draft_tokens: torch.Tensor, draft_token_ids: torch.Tensor, draft_probs: Optional[torch.Tensor], target_probs: torch.Tensor, sampling_metadata: vllm.v1.sample.metadata.SamplingMetadata, device: torch.device) -> torch.Tensor	14448
gptq_shuffle	torch::Tensor q_weight, torch::Tensor q_perm, int64_t bit	12660
matmul_ogs	(x, w, bias, routing_data: triton_kernels.routing.RoutingData | None = None, gather_indx: triton_kernels.routing.GatherIndx | None = None, scatter_indx: triton_kernels.routing.ScatterIndx | None = None, precision_config: triton_kernels.matmul_ogs.PrecisionConfig | None = None, betas: torch.Tensor | None = None, gammas: torch.Tensor | None = None, out_alpha: float | None = None, y: torch.Tensor | None = None, fused_activation: triton_kernels.matmul_ogs.FusedActivation | None = None, epilogue: triton_kernels.matmul_ogs.Epilogue | None = None)	8400
compute_aligned_M	(M: int, num_topk: int, local_num_experts: int, alignment: int, expert_tokens_meta: Optional[vllm.model_executor.layers.fused_moe.modular_kernel.ExpertTokensMetadata])	6600
compute_probs	(logits: torch.Tensor, cu_num_draft_tokens: torch.Tensor, sampling_metadata: vllm.v1.sample.metadata.SamplingMetadata) -> torch.Tensor	6600
check_aiter_fp8_linear_support	() -> bool	4900
process_fp8_weight_tensor_strategy	(weight: torch.Tensor, weight_scale: torch.Tensor, logical_widths: list[int], input_scale: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor]]	4828
convert_fp8		4813
_fp4_quantize	(A: torch.Tensor, A_scale: Optional[torch.Tensor], is_sf_swizzled_layout: bool) -> tuple[torch.Tensor, torch.Tensor]	4800
_fp8_perm	(m: torch.Tensor, idx: torch.Tensor) -> torch.Tensor	4800
_fp8_quantize	(A: torch.Tensor, A_scale: Optional[torch.Tensor], per_act_token: bool, block_shape: Optional[list[int]] = None) -> tuple[torch.Tensor, torch.Tensor]	4800
_int8_quantize	(A: torch.Tensor, A_scale: Optional[torch.Tensor], per_act_token: bool, block_shape: Optional[list[int]] = None) -> tuple[torch.Tensor, torch.Tensor]	4800
_maybe_pad_fp8_weight	(weight: torch.Tensor) -> torch.Tensor	4800
create_fp8_input_scale	(output_partition_sizes: list[int], weight_loader: Optional[Callable]) -> torch.nn.parameter.Parameter	4800
create_fp8_scale_parameter	(parameter_type: torch.nn.parameter.Parameter, output_partition_sizes: list[int], input_size_per_partition: int, block_size: Optional[list[int]], weight_loader: Optional[Callable]) -> torch.nn.parameter.Parameter	4800
create_fp8_weight_parameter	(output_size_per_partition: int, input_size_per_partition: int, weight_loader: Optional[Callable]) -> torch.nn.parameter.Parameter	4800
is_fp8	(x: Union[torch.dtype, torch.Tensor]) -> bool	4800
process_fp8_weight_channel_strategy	(weight: torch.Tensor, weight_scale: torch.Tensor, input_scale: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, typing.Optional[torch.Tensor]]	4800
free_shared_buffer	int64_t buffer	4510
shuffle_rows		4200
causal_conv1d_fn	(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], conv_states: torch.Tensor, query_start_loc: torch.Tensor, cache_indices: Optional[torch.Tensor] = None, has_initial_state: Optional[torch.Tensor] = None, activation: Optional[str] = 'silu', pad_slot_id: int = -1, metadata=None, validate_data=False)	3636
causal_conv1d_update	(x: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None, activation: Union[bool, str, NoneType] = None, cache_seqlens: Optional[torch.Tensor] = None, conv_state_indices: Optional[torch.Tensor] = None, num_accepted_tokens: Optional[torch.Tensor] = None, query_start_loc: Optional[torch.Tensor] = None, max_query_len: int = -1, pad_slot_id: int = -1, validate_data=False)	3636
selective_state_update	(state, x, dt, A, B, C, D=None, z=None, dt_bias=None, dt_softplus=False, state_batch_indices=None, pad_slot_id=-1, out=None)	3528
_lora_expand_fake	(inputs: torch.Tensor, lora_b_weights: list[torch.Tensor], output_tensor: torch.Tensor, token_lora_mapping: torch.Tensor, token_indices_sorted_by_lora_ids: torch.Tensor, num_tokens_per_lora: torch.Tensor, lora_token_start_loc: torch.Tensor, lora_ids: torch.Tensor, no_lora_flag_cpu: torch.Tensor, offset_start: int = 0, add_inputs: bool = False) -> None	2144
mla_decode_kvcache_cpu		576
TYPE_CHECKING		310
check_args	(q, k, v, o, varlen=True, max_seqlens=None, cu_seqlens_q=None, cu_seqlens_k=None)	100
weak_ref_tensor	torch::Tensor& tensor) { // Ensure tensor is on CUDA if (!tensor.is_cuda()) { throw std::runtime_error("Tensor must be on CUDA device"	28
convert_vertical_slash_indexes	torch::Tensor& block_count, // [BATCH, N_HEADS, NUM_ROWS] torch::Tensor& block_offset, // [BATCH, N_HEADS, NUM_ROWS, NNZ_S] torch::Tensor& column_count, // [BATCH, N_HEADS, NUM_ROWS] torch::Tensor& column_index, // [BATCH, N_HEADS, NUM_ROWS, NNZ_V] torch::Tensor q_seqlens, // [BATCH, ] torch::Tensor kv_seqlens, // [BATCH, ] torch::Tensor vertical_indexes, // [BATCH, N_HEADS, NNZ_V] torch::Tensor slash_indexes, // [BATCH, N_HEADS, NNZ_S] int64_t context_size, int64_t block_size_M, int64_t block_size_N, bool causal	13
convert_vertical_slash_indexes_mergehead	torch::Tensor& block_count, // [BATCH, N_HEADS, NUM_ROWS] torch::Tensor& block_offset, // [BATCH, N_HEADS, NUM_ROWS, NNZ_S] torch::Tensor& column_count, // [BATCH, N_HEADS, NUM_ROWS] torch::Tensor& column_index, // [BATCH, N_HEADS, NUM_ROWS, NNZ_V] torch::Tensor q_seqlens, // [BATCH, ] torch::Tensor kv_seqlens, // [BATCH, ] torch::Tensor vertical_indexes, // [BATCH, N_HEADS, NNZ_V] torch::Tensor slash_indexes, // [BATCH, N_HEADS, NNZ_S] torch::Tensor vertical_indices_count, // [N_HEADS, ] torch::Tensor slash_indices_count, int64_t context_size, int64_t block_size_M, int64_t block_size_N, bool causal	13
register_graph_buffers	fptr_t _fa, const std::vector<std::vector<int64_t>>& handles, const std::vector<std::vector<int64_t>>& offsets	6
_lora_shrink_fake	(inputs: torch.Tensor, lora_a_weights: list[torch.Tensor], output_tensor: torch.Tensor, token_lora_mapping: torch.Tensor, token_indices_sorted_by_lora_ids: torch.Tensor, num_tokens_per_lora: torch.Tensor, lora_token_start_loc: torch.Tensor, lora_ids: torch.Tensor, no_lora_flag_cpu: torch.Tensor, scaling: float) -> None	0
_mxfp4_quantize	(A: torch.Tensor, A_scale: Optional[torch.Tensor], per_act_token_quant: bool, block_shape: Optional[list[int]] = None) -> tuple[torch.Tensor, None]	0
_mxfp8_quantize	(A: torch.Tensor, A_scale: Optional[torch.Tensor], per_act_token_quant: bool, block_shape: Optional[list[int]] = None) -> tuple[torch.Tensor, torch.Tensor]	0
_validate_scale_shape	(a: torch.Tensor, a_scale: Optional[torch.Tensor], per_act_token_quant: bool, block_shape: Optional[list[int]]) -> None	0
allspark_repack_weight		0
apply_interleaved_rope	(x: torch.Tensor, mrope_section: list[int]) -> torch.Tensor	0
apply_repetition_penalties		0
apply_repetition_penalties_	torch::Tensor& logits, const torch::Tensor& prompt_mask, const torch::Tensor& output_mask, const torch::Tensor& repetition_penalties	0
apply_repetition_penalties_torch		0
apply_rotary	(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, seqlen_offsets: Union[int, torch.Tensor] = 0, cu_seqlens: Optional[torch.Tensor] = None, max_seqlen: Optional[int] = None, interleaved=False, inplace=False, conjugate=False) -> torch.Tensor	0
awq_dequantize	torch::Tensor _kernel, torch::Tensor _scaling_factors, torch::Tensor _zeros, int64_t split_k_iters, int64_t thx, int64_t thy	0
contextlib		0
correct_attn_out	(out: torch.Tensor, lses: torch.Tensor, cp_rank: int, ctx: vllm.attention.ops.common.CPTritonContext) -> tuple[torch.Tensor, torch.Tensor]	0
cp_lse_ag_out_rs	(cp_attn_out: torch.Tensor, cp_attn_lse: torch.Tensor, cp_group: vllm.distributed.parallel_state.GroupCoordinator, ctx: vllm.attention.ops.common.CPTritonContext = None)	0
CPUDNNLGEMMHandler		0
current_platform		0
dispatch_w8a8_blockscale_func	(use_cutlass: bool, use_aiter_and_is_supported: bool) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int], torch.dtype], torch.Tensor]	0
dispose	fptr_t _fa	0
envs		0
fast_plan_decode	(self, indptr_cpu: 'torch.Tensor', indices: 'torch.Tensor', last_page_len_cpu: 'torch.Tensor', seq_lens_cpu: 'torch.Tensor', num_qo_heads: 'int', num_kv_heads: 'int', head_dim: 'int', page_size: 'int', pos_encoding_mode: 'str' = 'NONE', window_left: 'int' = -1, logits_soft_cap: 'Optional[float]' = None, q_data_type: 'Optional[Union[str, torch.dtype]]' = 'float16', kv_data_type: 'Optional[Union[str, torch.dtype]]' = None, data_type: 'Optional[Union[str, torch.dtype]]' = None, sm_scale: 'Optional[float]' = None, rope_scale: 'Optional[float]' = None, rope_theta: 'Optional[float]' = None, non_blocking: 'bool' = True) -> 'None'	0
generate_uniform_probs	(num_tokens: int, num_draft_tokens: list[int], generators: dict[int, torch._C.Generator], device: torch.device) -> torch.Tensor	0
ggml_dequantize	torch::Tensor W, int64_t type, int64_t m, int64_t n, std::optional<at::ScalarType> const& dtype	0
hadacore_transform	torch::Tensor& x, bool inplace	0
init_custom_ar	const std::vector<int64_t>& fake_ipc_ptrs, torch::Tensor& rank_data, int64_t rank, bool fully_connected	0
init_custom_qr	int64_t rank, int64_t world_size, std::optional<int64_t> qr_max_size = std::nullopt	0
init_logger		0
input_guard	(fn: Callable[..., torch.Tensor]) -> Callable[..., torch.Tensor]	0
is_weak_contiguous	(x: torch.Tensor)	0
layernorm_fn	(x, weight, bias, z=None, eps=1e-06, group_size=None, norm_before_gate=True, is_rms_norm=False)	0
LLMM1		0
logger		0
normalize_batched_scales_shape	(scales: Optional[torch.Tensor], num_experts: int) -> Optional[torch.Tensor]	0
normalize_scales_shape	(scales: Optional[torch.Tensor]) -> Optional[torch.Tensor]	0
on_gfx1x	(*args, **kwargs)	0
open_mem_handle	torch::Tensor& mem_handle	0
Optional		0
qr_destroy	fptr_t _fa	0
qr_open_handles	fptr_t _fa, const std::vector<torch::Tensor>& handles	0
register_buffer	fptr_t _fa, const std::vector<int64_t>& fake_ipc_ptrs	0
register_fake		0
rejection_sample	(draft_token_ids: torch.Tensor, num_draft_tokens: list[int], max_spec_len: int, cu_num_draft_tokens: torch.Tensor, draft_probs: Optional[torch.Tensor], target_probs: torch.Tensor, bonus_token_ids: torch.Tensor, sampling_metadata: vllm.v1.sample.metadata.SamplingMetadata) -> torch.Tensor	0
requant_weight_ue8m0_inplace	(weight: torch.Tensor, weight_scale: torch.Tensor, block_size: collections.abc.Sequence[int] = (128, 128)) -> None	0
rmsnorm_fn	(x, weight, bias, z=None, eps=1e-06, group_size=None, norm_before_gate=True)	0
ScalarType		0
solve_tril	(A: torch.Tensor, cu_seqlens: Optional[torch.Tensor] = None, output_dtype: torch.dtype = torch.float32) -> torch.Tensor	0
torch		0
trtllm_prefill_attn_kvfp8_dequant	(kv_cache: 'torch.Tensor', block_tables_prefill: 'torch.Tensor', k_scale: 'torch.Tensor', v_scale: 'torch.Tensor', dequant_dtype: 'torch.dtype') -> 'tuple[torch.Tensor, torch.Tensor]'	0
Union		0
wvSplitK		0
wvSplitKQ		0
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		