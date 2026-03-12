名称	含义	代码路径	设备
    2 "weak_ref_tensor"	"Creates a weak reference to a CUDA tensor"	"csrc/ops.h"	"GPU"
    3 "paged_attention_v1"	"Computes attention for transformer models with paged KV cache"	"csrc/ops.h"	"GPU"
    4 "paged_attention_v2"	"Computes partitioned attention with paged KV cache"	"csrc/ops.h"	"GPU"
    5 "merge_attn_states"	"Merges attention states from different chunks"	"csrc/ops.h"	"GPU"
    6 "convert_vertical_slash_indexes"	"Converts vertical slash index format"	"csrc/ops.h"	"GPU"
    7 "convert_vertical_slash_indexes_mergehead"	"Converts vertical slash indexes with merge head"	"csrc/ops.h"	"GPU"
    8 "rms_norm"	"Root mean square normalization"	"csrc/ops.h"	"GPU"
    9 "fused_add_rms_norm"	"Fused add and RMS normalization"	"csrc/ops.h"	"GPU"
   10 "fused_qk_norm_rope"	"Fused Q/K normalization with rotary positional encoding"	"csrc/ops.h"	"GPU"
   11 "apply_repetition_penalties_"	"Applies repetition penalties to logits"	"csrc/ops.h"	"GPU"
   12 "top_k_per_row_prefill"	"Selects top-K values per row during prefill"	"csrc/ops.h"	"GPU"
   13 "top_k_per_row_decode"	"Selects top-K values per row during decode"	"csrc/ops.h"	"GPU"
   14 "rms_norm_static_fp8_quant"	"RMS normalization with static FP8 quantization"	"csrc/ops.h"	"GPU"
   15 "fused_add_rms_norm_static_fp8_quant"	"Fused add and RMS normalization with static FP8 quantization"	"csrc/ops.h"	"GPU"
   16 "rms_norm_dynamic_per_token_quant"	"RMS normalization with dynamic per-token quantization"	"csrc/ops.h"	"GPU"
   17 "rms_norm_per_block_quant"	"RMS normalization with per-block quantization"	"csrc/ops.h"	"GPU"
   18 "rotary_embedding"	"Computes rotary positional embeddings"	"csrc/ops.h"	"GPU"
   19 "silu_and_mul"	"SiLU activation followed by element-wise multiplication"	"csrc/ops.h"	"GPU"
   20 "silu_and_mul_quant"	"SiLU activation followed by element-wise multiplication with quantization"	"csrc/ops.h"	"GPU"
   21 "silu_and_mul_nvfp4_quant"	"SiLU activation followed by element-wise multiplication with NVFP4 quantization"	"csrc/ops.h"	"GPU"
   22 "persistent_masked_m_silu_mul_quant"	"Persistent masked SiLU and multiplication with quantization"	"csrc/ops.h"	"GPU"
   23 "mul_and_silu"	"Element-wise multiplication followed by SiLU activation"	"csrc/ops.h"	"GPU"
   24 "gelu_and_mul"	"GELU activation followed by element-wise multiplication"	"csrc/ops.h"	"GPU"
   25 "gelu_tanh_and_mul"	"GELU-tanh activation followed by element-wise multiplication"	"csrc/ops.h"	"GPU"
   26 "fatrelu_and_mul"	"FatReLU activation followed by element-wise multiplication"	"csrc/ops.h"	"GPU"
   27 "swigluoai_and_mul"	"SwiGLU OAI activation followed by element-wise multiplication"	"csrc/ops.h"	"GPU"
   28 "gelu_new"	"New GELU activation"	"csrc/ops.h"	"GPU"
   29 "gelu_fast"	"Fast GELU activation"	"csrc/ops.h"	"GPU"
   30 "gelu_quick"	"Quick GELU activation"	"csrc/ops.h"	"GPU"
   31 "cutlass_mla_decode"	"Cutlass MLA (Multihead Latent Attention) decode operation"	"csrc/ops.h"	"GPU"
   32 "get_cuda_view_from_cpu_tensor"	"Gets CUDA view from CPU tensor"	"csrc/ops.h"	"CPU/GPU"
   33 "awq_gemm"	"AWQ (Activation-aware Weight Quantization) GEMM operation"	"csrc/ops.h"	"GPU"
   34 "awq_dequantize"	"AWQ dequantization"	"csrc/ops.h"	"GPU"
   35 "permute_cols"	"Column permutation"	"csrc/ops.h"	"GPU"
   36 "ggml_dequantize"	"GGML dequantization"	"csrc/ops.h"	"GPU"
   37 "ggml_mul_mat_vec_a8"	"GGML matrix-vector multiplication with int8 A8 format"	"csrc/ops.h"	"GPU"
   38 "ggml_mul_mat_a8"	"GGML matrix-matrix multiplication with int8 A8 format"	"csrc/ops.h"	"GPU"
   39 "ggml_moe_a8"	"GGML Mixture-of-Experts with int8 A8 format"	"csrc/ops.h"	"GPU"
   40 "ggml_moe_a8_vec"	"GGML Mixture-of-Experts vector operation with int8 A8 format"	"csrc/ops.h"	"GPU"
   41 "ggml_moe_get_block_size"	"Gets GGML MoE block size"	"csrc/ops.h"	"CPU"
   42 "cutlass_scaled_mm_supports_fp4"	"Checks if CUTLASS scaled MM supports FP4"	"csrc/ops.h"	"CPU"
   43 "cutlass_scaled_mm_supports_fp8"	"Checks if CUTLASS scaled MM supports FP8"	"csrc/ops.h"	"CPU"
   44 "cutlass_scaled_mm_supports_block_fp8"	"Checks if CUTLASS scaled MM supports block FP8"	"csrc/ops.h"	"CPU"
   45 "cutlass_group_gemm_supported"	"Checks if CUTLASS group GEMM is supported"	"csrc/ops.h"	"CPU"
   46 "cutlass_scaled_fp4_mm"	"CUTLASS scaled matrix multiplication with FP4"	"csrc/ops.h"	"GPU"
   47 "cutlass_scaled_mm"	"CUTLASS scaled matrix multiplication"	"csrc/ops.h"	"GPU"
   48 "cutlass_moe_mm"	"CUTLASS Mixture-of-Experts matrix multiplication"	"csrc/ops.h"	"GPU"
   49 "cutlass_fp4_group_mm"	"CUTLASS FP4 grouped matrix multiplication"	"csrc/ops.h"	"GPU"
   50 "get_cutlass_moe_mm_data"	"Gets CUTLASS MoE MM data"	"csrc/ops.h"	"CPU"
   51 "get_cutlass_moe_mm_problem_sizes"	"Gets CUTLASS MoE MM problem sizes"	"csrc/ops.h"	"CPU"
   52 "get_cutlass_pplx_moe_mm_data"	"Gets CUTLASS PPLX MoE MM data"	"csrc/ops.h"	"CPU"
   53 "cutlass_scaled_mm_azp"	"CUTLASS scaled MM with zero-point adjustment"	"csrc/ops.h"	"GPU"
   54 "cutlass_sparse_scaled_mm_supported"	"Checks if CUTLASS sparse scaled MM is supported"	"csrc/ops.h"	"CPU"
   55 "cutlass_scaled_sparse_mm"	"CUTLASS scaled sparse matrix multiplication"	"csrc/ops.h"	"GPU"
   56 "cutlass_sparse_compress"	"CUTLASS sparse matrix compression"	"csrc/ops.h"	"GPU"
   57 "scaled_fp4_quant"	"Scaled FP4 quantization"	"csrc/ops.h"	"GPU"
   58 "scaled_fp4_experts_quant"	"Scaled FP4 expert quantization"	"csrc/ops.h"	"GPU"
   59 "per_token_group_quant_fp8"	"Per-token group quantization with FP8"	"csrc/ops.h"	"GPU"
   60 "per_token_group_quant_int8"	"Per-token group quantization with INT8"	"csrc/ops.h"	"GPU"
   61 "per_token_group_quant_8bit_packed"	"Per-token group quantization with 8-bit packed"	"csrc/ops.h"	"GPU"
   62 "static_scaled_int8_quant"	"Static scaled INT8 quantization"	"csrc/ops.h"	"GPU"
   63 "dynamic_scaled_int8_quant"	"Dynamic scaled INT8 quantization"	"csrc/ops.h"	"GPU"
   64 "gptq_gemm"	"GPTQ GEMM operation"	"csrc/ops.h"	"GPU"
   65 "gptq_shuffle"	"GPTQ weight shuffling"	"csrc/ops.h"	"GPU"
   66 "static_scaled_fp8_quant"	"Static scaled FP8 quantization"	"csrc/ops.h"	"GPU"
   67 "dynamic_scaled_fp8_quant"	"Dynamic scaled FP8 quantization"	"csrc/ops.h"	"GPU"
   68 "dynamic_per_token_scaled_fp8_quant"	"Dynamic per-token scaled FP8 quantization"	"csrc/ops.h"	"GPU"
   69 "selective_scan_fwd"	"Forward selective scan for Mamba models"	"csrc/ops.h"	"GPU"
   70 "dynamic_4bit_int_moe_cpu"	"Dynamic 4-bit integer MoE on CPU"	"csrc/ops.h"	"CPU"
   71 "init_custom_ar"	"Initializes custom all-reduce"	"csrc/ops.h"	"CPU"
   72 "all_reduce"	"Performs all-reduce operation"	"csrc/ops.h"	"GPU"
   73 "dispose"	"Disposes custom all-reduce handle"	"csrc/ops.h"	"CPU"
   74 "meta_size"	"Gets metadata size"	"csrc/ops.h"	"CPU"
   75 "register_buffer"	"Registers buffer for custom all-reduce"	"csrc/ops.h"	"CPU"
   76 "get_graph_buffer_ipc_meta"	"Gets graph buffer IPC metadata"	"csrc/ops.h"	"CPU"
   77 "register_graph_buffers"	"Registers graph buffers"	"csrc/ops.h"	"CPU"
   78 "allocate_shared_buffer_and_handle"	"Allocates shared buffer and handle"	"csrc/ops.h"	"CPU"
   79 "open_mem_handle"	"Opens memory handle"	"csrc/ops.h"	"CPU"
   80 "free_shared_buffer"	"Frees shared buffer"	"csrc/ops.h"	"CPU"
   81 "hadacore_transform"	"HadaCore transformation"	"csrc/ops.h"	"GPU"
   82 "init_custom_qr"	"Initializes custom QR (ROCm)"	"csrc/ops.h"	"GPU"
   83 "qr_destroy"	"Destroys custom QR handle (ROCm)"	"csrc/ops.h"	"CPU"
   84 "qr_get_handle"	"Gets QR handle (ROCm)"	"csrc/ops.h"	"CPU"
   85 "qr_open_handles"	"Opens QR handles (ROCm)"	"csrc/ops.h"	"CPU"
   86 "qr_all_reduce"	"QR all-reduce (ROCm)"	"csrc/ops.h"	"GPU"
   87 "qr_max_size"	"Gets QR max size (ROCm)"	"csrc/ops.h"	"CPU"
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			
			