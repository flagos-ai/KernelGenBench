# cuBLAS Baseline 海光 DCU 测试记录

## 环境
- 硬件: Hygon DCU (hipBLAS)
- Runtime: flagbench.runtime → HygonBackend
- 日期: 2025-07-13
- 更新: 2026-04-14 (修复代码bug + 枚举映射 + data type映射)

## 测试结果 (50个baseline)

### ✅ 通过 (42个)

| # | Baseline | 备注 |
|---|----------|------|
| 1 | cublasSgemm_v2 | |
| 2 | cublasDaxpy_v2 | |
| 3 | cublasSgemmStridedBatched | |
| 4 | cublasSgeam | |
| 5 | cublasDgemmBatched | |
| 6 | cublasSdot_v2 | gemm fallback |
| 7 | cublasSgemvBatched | |
| 8 | cublasDasum_v2 | gemm fallback |
| 9 | cublasCdotu_v2 | gemm fallback |
| 10 | cublasZdotc_v2 | gemm fallback |
| 11 | cublasCcopy_v2 | |
| 12 | cublasCgeru_v2 | |
| 13 | cublasZgerc_v2 | |
| 14 | cublasDgemv_v2 | torch fallback |
| 15 | cublasCgemv_v2 | torch fallback |
| 16 | cublasCgemmStridedBatched_64 | |
| 17 | cublasCgemmStridedBatched | |
| 18 | cublasCgemvStridedBatched | |
| 19 | cublasCsymm_v2 | |
| 20 | cublasCsymv_v2 | |
| 21 | cublasDcopy_v2 | |
| 22 | cublasDgemmStridedBatched_64 | |
| 23 | cublasDgemmStridedBatched | |
| 24 | cublasDgemvBatched | |
| 25 | cublasDtrsmBatched | |
| 26 | cublasHgemmStridedBatched | |
| 27 | cublasSaxpy_v2 | |
| 28 | cublasSgemmBatched_64 | |
| 29 | cublasSger_v2 | |
| 30 | cublasSscal_v2 | |
| 31 | cublasZgemmBatched | |
| 32 | cublasZgemvStridedBatched | |
| 33 | cublasZswap_v2 | |
| 34 | cublasCgemvBatched_64 | 修复: 全局变量名 |
| 35 | cublasHgemmBatched | 修复: c_half移到模块级 |
| 36 | cublasSdgmm | 修复: map_diag→map_side |
| 37 | cublasZgemmStridedBatched | 修复: CUBLAS_OP_N定义 |
| 38 | cublasCsyrkEx | 修复: 测试列主序 + data type映射 |
| 39 | cublasDgemvStridedBatched | 修复: map_op移到isinstance外 |
| 40 | cublasSgemvStridedBatched | 修复: map_op移到isinstance外 |
| 41 | cublasZgemvBatched | 修复: map_op移到isinstance外 |
| 42 | cublasSgemmEx | 修复: map_op顺序 + data type映射 |

### ❌ 数值精度问题 (8个，海光特有，NV上均通过)

| # | Baseline | 错误 |
|---|----------|------|
| 1 | cublasCgemm_v2 | rel diff 7.87 (torch fallback) |
| 2 | cublasDsbmv_v2 | rel diff 10.24 |
| 3 | cublasDsyr2_v2 | rel diff 3.39 |
| 4 | cublasSsyrk_v2 | rel diff 26.78 |
| 5 | cublasStbmv_v2 | rel diff 9.79 |
| 6 | cublasStrsm_v2 | rel diff 0.71 |
| 7 | cublasStrsv_v2 | rel diff 0.84 |
| 8 | cublasZtrsmBatched | rel diff 0.96 (枚举已修复，精度不过) |
