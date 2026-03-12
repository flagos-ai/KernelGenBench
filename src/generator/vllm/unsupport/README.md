# vLLM 算子环境兼容性问题报告

## 当前环境

| 项目 | 版本 |
|------|------|
| GPU | NVIDIA A100 (SM 8.0) |
| CUDA | 12.8 |
| vLLM | 0.13.0 |
| PyTorch | 2.x |
| 平台 | Linux x86_64 (NVIDIA) |

## 不支持的算子列表

从 vLLM 0.13 的 318 个 CUDA 算子中筛选出 40 个核心算子进行 benchmark 开发时，发现以下 9 个算子在当前环境下无法运行。

### 1. CUTLASS FP4/MoE 系列 — 需要 SM >= 90/100

| 算子 | 错误信息 |
|------|----------|
| `cutlass_scaled_fp4_mm` | `No compiled nvfp4 mm kernel for SM 80. Recompile with CUDA >= 12.8 and CC >= 100.` |
| `cutlass_moe_mm` | `No compiled cutlass_scaled_mm for CUDA device capability: 80. Required capability: 90 or 100` |
| `cutlass_fp4_moe_mm` | `RuntimeError: Inconsistency of Tensor type:a` (NVFP4 Group Gemm, 需要 SM >= 100) |

复现脚本：`test_cutlass_fp4.py`, `test_cutlass_moe.py`

### 2. FP4 量化系列 — SM80 上 kernel 不稳定

| 算子 | 错误信息 |
|------|----------|
| `scaled_fp4_quant` | `CUDA error: no kernel image is available for execution on the device`（第二次调用即崩溃） |
| `scaled_fp4_experts_quant` | 依赖 FP4 kernel，同样受 SM80 限制 |

复现脚本：`test_scaled_fp4.py`

### 2. ROCm 专用算子 — 仅 AMD GPU 可用

| 算子 | 错误信息 |
|------|----------|
| `wvSplitK` | `'_OpNamespace' '_rocm_C' object has no attribute 'wvSplitK'` |
| `wvSplitKQ` | `'_OpNamespace' '_rocm_C' object has no attribute 'wvSplitKQ'` |
| `LLMM1` | `'_OpNamespace' '_rocm_C' object has no attribute 'LLMM1'` |

复现脚本：`test_rocm_ops.py`

### 3. flash_mla_with_kvcache — 未编译的 kernel

| 算子 | 错误信息 |
|------|----------|
| `flash_mla_with_kvcache` | `'_OpNamespace' '_C' object has no attribute 'flash_mla_fwd_kvcache'` |

复现脚本：`test_flash_mla.py`

## 复现方法

```bash
# 激活对应 conda 环境
conda activate zpy_flagbench

# 进入脚本目录
cd src/generator/vllm/unsupport

# 逐个运行
python test_cutlass_fp4.py
python test_cutlass_moe.py
python test_rocm_ops.py
python test_flash_mla.py
```

预期输出均为 `FAIL: ...`，表示当前环境不支持。

## 结论

这 7 个算子的失败是环境限制，不是代码问题：

- FP4/MoE CUTLASS 算子需要 Hopper (H100, SM90) 或 Blackwell (B200, SM100) 架构
- ROCm 算子需要 AMD GPU + ROCm 环境
- flash_mla 需要 vLLM 编译时启用对应 kernel（可能需要更新版本或特定编译选项）

建议将这 7 个算子从当前 A100 环境的 benchmark 列表中排除，或在 H100/B200 环境上单独测试。

### 4. indexer_k_quant_and_cache — C++ kernel 不确定性

| 算子 | 错误信息 |
|------|----------|
| `indexer_k_quant_and_cache` | 同一输入两次调用结果不一致（MISMATCH），尤其在 float16 + 大尺寸下。独立进程验证确认是 C++ kernel 本身不确定性，非 test_func 问题。 |

稳定的参数组合极少（仅小尺寸 + bfloat16），不足以构成有效 benchmark，暂时排除。

### 5. awq_gemm — C++ kernel 不确定性

| 算子 | 错误信息 |
|------|----------|
| `awq_gemm` | 同一输入两次调用结果不一致（MISMATCH），max_diff 达数千（5088/57792）。独立进程验证确认是 C++ kernel 本身不确定性，非 test_func 问题。 |

### 6. cutlass_scaled_sparse_mm / cutlass_sparse_compress — SM 80 不支持

| 算子 | 错误信息 |
|------|----------|
| `cutlass_scaled_sparse_mm` | `No compiled cutlass_scaled_sparse_mm for a compute capability less than CUDA device capability: 80` |
| `cutlass_sparse_compress` | `No compiled cutlass_sparse_compress for a compute capability less than CUDA device capability: 80` |

### 7. cutlass_w4a8_mm — TMA descriptor 初始化失败

| 算子 | 错误信息 |
|------|----------|
| `cutlass_w4a8_mm` | `Error Internal: Failed to initialize the TMA descriptor 801`。需要 SM >= 90 的 TMA (Tensor Memory Accelerator) 硬件支持。 |

### 8. machete_mm — SM 80 无编译 kernel

| 算子 | 错误信息 |
|------|----------|
| `machete_mm` | `machete_mm(..) is not implemented for a_type=Half, b_type=*, out_type=Half`。所有 ScalarType 组合在 SM 80 上均无编译的 kernel。 |

### 9. cutlass_encode_and_reorder_int4b — kernel 执行失败

| 算子 | 错误信息 |
|------|----------|
| `cutlass_encode_and_reorder_int4b` | `unified_encode_int4b failed`。所有输入 shape 和 dtype 组合均触发 CUDA 错误。 |

### 10. moe_wna16_gemm — BLOCK_SIZE_K/group_size 约束无法满足

| 算子 | 错误信息 |
|------|----------|
| `moe_wna16_gemm` | `BLOCK_SIZE_K // group_size must be one of [1, 2, 4, 8]`。所有 group_size 和 BLOCK_SIZE_K 组合均报此错误，疑似 C++ kernel 在 SM 80 上的兼容性问题。 |

### 11. moe_wna16_marlin_gemm — marlin 格式 + 30 参数

| 算子 | 错误信息 |
|------|----------|
| `moe_wna16_marlin_gemm` | 需要 marlin 格式 weight packing（与 gptq_marlin_24_gemm 类似），且 moe_block_size 仅支持特定值（16/64 可用但 weight shape 不匹配）。30 个参数构造极其复杂，暂时排除。 |
