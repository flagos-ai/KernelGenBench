# from .run import TestRunner
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .dataset.kernel_list import PYTORCH_OPERATORS
from sandbox.register import Register, register, REGISTERED_OPS
from sandbox.verifier.test_parametrize import Param, parametrize, label

# Import baseline modules to trigger registration
try:
    from .dataset.baseline import cupy as cupy_baseline
except ImportError:
    cupy_baseline = None

try:
    from .dataset.baseline import cublas as cublas_baseline
except ImportError:
    cublas_baseline = None


import os
DISPATCH_TORCH_LIB = os.environ.get("DISPATCH_TORCH_LIB", "1") == "1"
# __version__ = "0.1"
# device = runtime.device.name
# vendor_name = runtime.device.vendor_name
aten_lib = torch.library.Library("aten", "IMPL")

current_work_registrar = None

def enable(config, lib=aten_lib, unused=None, registrar=Register):
    global current_work_registrar
    if not DISPATCH_TORCH_LIB:
        current_work_registrar = None
        return
    current_work_registrar = registrar(
        config=config,
        user_unused_ops_list=[] if unused is None else unused,
        lib=lib,
    )


class use_gems:
    def __init__(self, config, unused=None):
        self.lib = torch.library.Library("aten", "IMPL")
        self.unused = [] if unused is None else unused
        self.registrar = Register
        self.config = config

    def __enter__(self):
        enable(config=self.config, lib=self.lib, unused=self.unused, registrar=self.registrar)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global current_work_registrar
        del self.lib
        del self.unused
        del self.registrar
        del current_work_registrar


def all_ops():
    return current_work_registrar.get_all_ops()

accuracy_modules = [
    "flagbench.accuracy.test_attention_ops",
    "flagbench.accuracy.test_binary_pointwise_ops",
    "flagbench.accuracy.test_blas_ops",
    "flagbench.accuracy.test_distribution_ops",
    "flagbench.accuracy.test_general_reduction_ops",
    "flagbench.accuracy.test_norm_ops",
    # "flagbench.accuracy.test_pointwise_type_promotion",
    "flagbench.accuracy.test_reduction_ops",
    "flagbench.accuracy.test_special_ops",
    "flagbench.accuracy.test_tensor_constructor_ops",
    "flagbench.accuracy.test_unary_pointwise_ops",
    "flagbench.accuracy.test_non_flaggems_ops",
    "flagbench.accuracy.test_v2_ops",
    "flagbench.accuracy.test_v2_1_ops",
    "flagbench.accuracy.test_v2_1_ops_with_benchmark",
    # "flagbench.accuracy.test_non_torch_prelu_ops",  # Temporarily disabled - missing example baseline __init__.py
    "flagbench.accuracy.cupy.test_caxpy_cublas_ops",
    "flagbench.accuracy.cupy.test_cdgmm_cublas_ops",
    "flagbench.accuracy.cupy.test_cdotc_cublas_ops",
    "flagbench.accuracy.cupy.test_cdotu_cublas_ops",
    "flagbench.accuracy.cupy.test_cgeam_cublas_ops",
    "flagbench.accuracy.cupy.test_cgemm_cublas_ops",
    "flagbench.accuracy.cupy.test_cgemv_cublas_ops",
    "flagbench.accuracy.cupy.test_cgerc_cublas_ops",
    "flagbench.accuracy.cupy.test_cgeru_cublas_ops",
    "flagbench.accuracy.cupy.test_cscal_cublas_ops",
    "flagbench.accuracy.cupy.test_csyrk_cublas_ops",
    "flagbench.accuracy.cupy.test_dasum_cublas_ops",
    "flagbench.accuracy.cupy.test_daxpy_cublas_ops",
    "flagbench.accuracy.cupy.test_ddgmm_cublas_ops",
    "flagbench.accuracy.cupy.test_ddot_cublas_ops",
    "flagbench.accuracy.cupy.test_dgeam_cublas_ops",
    "flagbench.accuracy.cupy.test_dgemm_cublas_ops",
    "flagbench.accuracy.cupy.test_dgemv_cublas_ops",
    "flagbench.accuracy.cupy.test_dger_cublas_ops",
    "flagbench.accuracy.cupy.test_dnrm2_cublas_ops",
    "flagbench.accuracy.cupy.test_dsbmv_cublas_ops",
    "flagbench.accuracy.cupy.test_dscal_cublas_ops",
    "flagbench.accuracy.cupy.test_dsyrk_cublas_ops",
    "flagbench.accuracy.cupy.test_hgemm_cublas_ops",
    "flagbench.accuracy.cupy.test_sasum_cublas_ops",
    "flagbench.accuracy.cupy.test_saxpy_cublas_ops",
    "flagbench.accuracy.cupy.test_sdgmm_cublas_ops",
    "flagbench.accuracy.cupy.test_sdot_cublas_ops",
    "flagbench.accuracy.cupy.test_sgeam_cublas_ops",
    "flagbench.accuracy.cupy.test_sgemm_cublas_ops",
    "flagbench.accuracy.cupy.test_sgemv_cublas_ops",
    "flagbench.accuracy.cupy.test_sger_cublas_ops",
    "flagbench.accuracy.cupy.test_snrm2_cublas_ops",
    "flagbench.accuracy.cupy.test_ssbmv_cublas_ops",
    "flagbench.accuracy.cupy.test_sscal_cublas_ops",
    "flagbench.accuracy.cupy.test_ssyrk_cublas_ops",
    "flagbench.accuracy.cupy.test_zaxpy_cublas_ops",
    "flagbench.accuracy.cupy.test_zdgmm_cublas_ops",
    "flagbench.accuracy.cupy.test_zdotc_cublas_ops",
    "flagbench.accuracy.cupy.test_zdotu_cublas_ops",
    "flagbench.accuracy.cupy.test_zgeam_cublas_ops",
    "flagbench.accuracy.cupy.test_zgemm_cublas_ops",
    "flagbench.accuracy.cupy.test_zgemv_cublas_ops",
    "flagbench.accuracy.cupy.test_zgerc_cublas_ops",
    "flagbench.accuracy.cupy.test_zgeru_cublas_ops",
    "flagbench.accuracy.cupy.test_zscal_cublas_ops",
    "flagbench.accuracy.cupy.test_zsyrk_cublas_ops",
    # cuBLAS ctypes operators (10 ops)
    "flagbench.accuracy.cublas.test_cublasSaxpy_v2",
    "flagbench.accuracy.cublas.test_cublasSscal_v2",
    "flagbench.accuracy.cublas.test_cublasSgemmStridedBatched",
    "flagbench.accuracy.cublas.test_cublasDgemmStridedBatched",
    "flagbench.accuracy.cublas.test_cublasZgemmStridedBatched",
    "flagbench.accuracy.cublas.test_cublasHgemmStridedBatched",
    "flagbench.accuracy.cublas.test_cublasSgemvStridedBatched",
    "flagbench.accuracy.cublas.test_cublasDgemvStridedBatched",
    "flagbench.accuracy.cublas.test_cublasCgemvStridedBatched",
    "flagbench.accuracy.cublas.test_cublasZgemvStridedBatched",
    # 40 new cuBLAS operators
    "flagbench.accuracy.cublas.test_cublasCcopy_v2",
    "flagbench.accuracy.cublas.test_cublasCdotu_v2",
    "flagbench.accuracy.cublas.test_cublasCgemmStridedBatched",
    "flagbench.accuracy.cublas.test_cublasCgemmStridedBatched_64",
    "flagbench.accuracy.cublas.test_cublasCgemm_v2",
    "flagbench.accuracy.cublas.test_cublasCgemvBatched_64",
    "flagbench.accuracy.cublas.test_cublasCgemv_v2",
    "flagbench.accuracy.cublas.test_cublasCgeru_v2",
    "flagbench.accuracy.cublas.test_cublasCsymm_v2",
    "flagbench.accuracy.cublas.test_cublasCsymv_v2",
    "flagbench.accuracy.cublas.test_cublasCsyrkEx",
    "flagbench.accuracy.cublas.test_cublasDasum_v2",
    "flagbench.accuracy.cublas.test_cublasDaxpy_v2",
    "flagbench.accuracy.cublas.test_cublasDcopy_v2",
    "flagbench.accuracy.cublas.test_cublasDgemmBatched",
    "flagbench.accuracy.cublas.test_cublasDgemmStridedBatched_64",
    "flagbench.accuracy.cublas.test_cublasDgemvBatched",
    "flagbench.accuracy.cublas.test_cublasDgemv_v2",
    "flagbench.accuracy.cublas.test_cublasDsbmv_v2",
    "flagbench.accuracy.cublas.test_cublasDsyr2_v2",
    "flagbench.accuracy.cublas.test_cublasDtrsmBatched",
    "flagbench.accuracy.cublas.test_cublasHgemmBatched",
    "flagbench.accuracy.cublas.test_cublasSdgmm",
    "flagbench.accuracy.cublas.test_cublasSdot_v2",
    "flagbench.accuracy.cublas.test_cublasSgeam",
    "flagbench.accuracy.cublas.test_cublasSgemmBatched_64",
    "flagbench.accuracy.cublas.test_cublasSgemmEx",
    "flagbench.accuracy.cublas.test_cublasSgemm_v2",
    "flagbench.accuracy.cublas.test_cublasSgemvBatched",
    "flagbench.accuracy.cublas.test_cublasSger_v2",
    "flagbench.accuracy.cublas.test_cublasSsyrk_v2",
    "flagbench.accuracy.cublas.test_cublasStbmv_v2",
    "flagbench.accuracy.cublas.test_cublasStrsm_v2",
    "flagbench.accuracy.cublas.test_cublasStrsv_v2",
    "flagbench.accuracy.cublas.test_cublasZdotc_v2",
    "flagbench.accuracy.cublas.test_cublasZgemmBatched",
    "flagbench.accuracy.cublas.test_cublasZgemvBatched",
    "flagbench.accuracy.cublas.test_cublasZgerc_v2",
    "flagbench.accuracy.cublas.test_cublasZswap_v2",
    "flagbench.accuracy.cublas.test_cublasZtrsmBatched",
    # vLLM operators (v0.15)
    "flagbench.accuracy.vllm15.test_moe_align_block_size",
    "flagbench.accuracy.vllm15.test_selective_scan_fwd",
    "flagbench.accuracy.vllm15.test_paged_attention_v1",
    "flagbench.accuracy.vllm15.test_scaled_fp8_quant",
    "flagbench.accuracy.vllm15.test_rotary_embedding",
    "flagbench.accuracy.vllm15.test_fused_add_rms_norm",
    "flagbench.accuracy.vllm15.test_topk_softmax",
    "flagbench.accuracy.vllm15.test_fused_qk_norm_rope",
    "flagbench.accuracy.vllm15.test_scaled_int8_quant",
    "flagbench.accuracy.vllm15.test_cutlass_scaled_mm",
    # vLLM operators (v0.13)
    "flagbench.accuracy.vllm13.test_moe_align_block_size",
    "flagbench.accuracy.vllm13.test_selective_scan_fwd",
    "flagbench.accuracy.vllm13.test_paged_attention_v1",
    "flagbench.accuracy.vllm13.test_scaled_fp8_quant",
    "flagbench.accuracy.vllm13.test_rotary_embedding",
    "flagbench.accuracy.vllm13.test_fused_add_rms_norm",
    "flagbench.accuracy.vllm13.test_topk_softmax",
    "flagbench.accuracy.vllm13.test_fused_qk_norm_rope",
    "flagbench.accuracy.vllm13.test_scaled_int8_quant",
    "flagbench.accuracy.vllm13.test_cutlass_scaled_mm",
    "flagbench.accuracy.vllm13.test_rms_norm",
    "flagbench.accuracy.vllm13.test_rms_norm_dynamic_per_token_quant",
    "flagbench.accuracy.vllm13.test_moe_sum",
    "flagbench.accuracy.vllm13.test_permute_cols",
    "flagbench.accuracy.vllm13.test_shuffle_rows",
    "flagbench.accuracy.vllm13.test_convert_fp8",
    "flagbench.accuracy.vllm13.test_hadacore_transform",
    "flagbench.accuracy.vllm13.test_swap_blocks",
    "flagbench.accuracy.vllm13.test_grouped_topk",
    "flagbench.accuracy.vllm13.test_rms_norm_per_block_quant",
    "flagbench.accuracy.vllm13.test_gptq_shuffle",
    "flagbench.accuracy.vllm13.test_merge_attn_states",
    "flagbench.accuracy.vllm13.test_reshape_and_cache",
    "flagbench.accuracy.vllm13.test_reshape_and_cache_flash",
    "flagbench.accuracy.vllm13.test_concat_and_cache_mla",
    "flagbench.accuracy.vllm13.test_paged_attention_v2",
    "flagbench.accuracy.vllm13.test_gptq_gemm",
    "flagbench.accuracy.vllm13.test_cutlass_pack_scale_fp8",
    "flagbench.accuracy.vllm13.test_cutlass_scaled_mm_azp",
    "flagbench.accuracy.vllm13.test_gptq_marlin_24_gemm",
    "flagbench.accuracy.vllm13.test_copy_blocks",
    "flagbench.accuracy.vllm13.test_copy_blocks_mla",
    "flagbench.accuracy.vllm13.test_allspark_w8a16_gemm",
    "flagbench.accuracy.vllm13.test_gather_and_maybe_dequant_cache",
    "flagbench.accuracy.vllm13.test_convert_vertical_slash_indexes",
    "flagbench.accuracy.vllm13.test_cp_gather_cache",
    "flagbench.accuracy.vllm13.test_cp_gather_indexer_k_quant_cache",
    "flagbench.accuracy.vllm13.test_marlin_int4_fp8_preprocess",
    "flagbench.accuracy.vllm13.test_gptq_marlin_gemm",
    "flagbench.accuracy.vllm13.test_ggml_mul_mat_a8",
    "flagbench.accuracy.vllm13.test_ggml_mul_mat_vec_a8",
    "flagbench.accuracy.vllm13.test_gptq_marlin_moe_repack",
    "flagbench.accuracy.vllm13.test_awq_marlin_moe_repack",
    "flagbench.accuracy.vllm13.test_moe_lora_align_block_size",
    "flagbench.accuracy.vllm13.test_batched_moe_align_block_size",
    "flagbench.accuracy.vllm13.test_apply_repetition_penalties_cuda",
    "flagbench.accuracy.vllm13.test_awq_gemm",
    "flagbench.accuracy.vllm13.test_ggml_dequantize",
    "flagbench.accuracy.vllm13.test_ggml_moe_a8",
    "flagbench.accuracy.vllm13.test_ggml_moe_a8_vec",
]
perf_modules = [
    "flagbench.perfermance.test_attention_perf",
    "flagbench.perfermance.test_binary_pointwise_perf",
    "flagbench.perfermance.test_blas_perf",
    "flagbench.perfermance.test_distribution_perf",
    # skip fused for now
    # "flagbench.perfermance.test_fused_perf",
    "flagbench.perfermance.test_generic_pointwise_perf",
    "flagbench.perfermance.test_norm_perf",
    "flagbench.perfermance.test_reduction_perf",
    "flagbench.perfermance.test_select_and_slice_perf",
    "flagbench.perfermance.test_special_perf",
    "flagbench.perfermance.test_tensor_concat_perf",
    "flagbench.perfermance.test_tensor_constructor_perf",
    "flagbench.perfermance.test_unary_pointwise_perf",
]

__all__ = [
    "enable",
    "use_gems",
    "register",
    "accuracy_modules",
]
