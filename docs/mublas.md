# cuBLAS → MUBLAS 迁移对照

## 对照规则

| cuBLAS | MUBLAS | 规则 |
|--------|--------|------|
| `cublas` 前缀 | `mublas` 前缀 | `cu` → `mu` |
| `_v2` 后缀 | 去掉 | MUBLAS 无 `_v2` |
| `_64` 后缀 | 去掉 | MUBLAS 无 `_64`，参数用 `int` 非 `int64_t` |
| `cublasSgemmEx` | `mublasGemmEx` | 无类型前缀，单例泛型函数 |
| `cublasHandle_t` | `mublasHandle_t` | |
| `cudaStream_t` | `MUstream` | |
| `cuComplex` | `muComplex` | |
| `cuDoubleComplex` | `muDoubleComplex` | |
| `CUBLAS_OP_*` | `MUBLAS_OP_*` | |
| `CUBLAS_FILL_MODE_*` | `MUBLAS_FILL_MODE_*` | |
| `CUBLAS_DIAG_*` | `MUBLAS_DIAG_*` | |
| `CUBLAS_SIDE_*` | `MUBLAS_SIDE_*` | |
| `CUBLAS_STATUS_*` | `MUBLAS_STATUS_*` | |
| `<cublas_v2.h>` | `<mublas.h>` | |
| `-lcublas` | `-lmublas` | |

## 详细对照表

```
# ===== 函数替换 =====
cublasCcopy_v2              → mublasCcopy
cublasCdotu_v2              → mublasCdotu
cublasCgemmStridedBatched   → mublasCgemmStridedBatched
cublasCgemmStridedBatched_64→ mublasCgemmStridedBatched  (去 _64)
cublasCgemm_v2              → mublasCgemm
cublasCgemvBatched_64       → mublasCgemvBatched          (去 _64)
cublasCgemvStridedBatched   → mublasCgemvStridedBatched
cublasCgemv_v2              → mublasCgemv
cublasCgeru_v2              → mublasCgeru
cublasCsymm_v2              → mublasCsymm
cublasCsymv_v2              → mublasCsymv
cublasCsyrkEx               → mublasCsyrkEx
cublasDasum_v2              → mublasDasum
cublasDaxpy_v2              → mublasDaxpy
cublasDcopy_v2              → mublasDcopy
cublasDgemmBatched          → mublasDgemmBatched
cublasDgemmStridedBatched   → mublasDgemmStridedBatched
cublasDgemmStridedBatched_64→ mublasDgemmStridedBatched  (去 _64)
cublasDgemm_v2              → mublasDgemm
cublasDgemvBatched          → mublasDgemvBatched
cublasDgemvStridedBatched   → mublasDgemvStridedBatched
cublasDgemv_v2              → mublasDgemv
cublasDsbmv_v2              → mublasDsbmv
cublasDsyr2_v2              → mublasDsyr2
cublasDtrsmBatched          → mublasDtrsmBatched
cublasHgemmBatched          → mublasHgemmBatched
cublasHgemmStridedBatched   → mublasHgemmStridedBatched
cublasSaxpy_v2              → mublasSaxpy
cublasSdgmm                 → mublasSdgmm
cublasSdot_v2               → mublasSdot
cublasSgeam                 → mublasSgeam
cublasSgemmBatched_64       → mublasSgemmBatched          (去 _64)
cublasSgemmEx               → mublasGemmEx                (去类型前缀 S)
cublasSgemmStridedBatched   → mublasSgemmStridedBatched
cublasSgemm_v2              → mublasSgemm
cublasSgemvBatched          → mublasSgemvBatched
cublasSgemvStridedBatched   → mublasSgemvStridedBatched
cublasSger_v2               → mublasSger
cublasSscal_v2              → mublasSscal
cublasSsyrk_v2              → mublasSsyrk
cublasStbmv_v2              → mublasStbmv
cublasStrsm_v2              → mublasStrsm
cublasStrsv_v2              → mublasStrsv
cublasZdotc_v2              → mublasZdotc
cublasZgemmBatched          → mublasZgemmBatched
cublasZgemmStridedBatched   → mublasZgemmStridedBatched
cublasZgemvBatched          → mublasZgemvBatched
cublasZgemvStridedBatched   → mublasZgemvStridedBatched
cublasZgerc_v2              → mublasZgerc
cublasZswap_v2              → mublasZswap
cublasZtrsmBatched          → mublasZtrsmBatched

# ===== 类型替换 =====
cublasHandle_t              → mublasHandle_t
cublasStatus_t              → mublasStatus
cublasOperation_t           → mublasOperation_t / mublasOperation
cublasFillMode_t            → mublasFillMode_t
cublasDiagType_t            → mublasDiagType_t
cublasSideMode_t            → mublasSideMode_t
cudaStream_t                → MUstream
cuComplex                   → muComplex
cuDoubleComplex             → muDoubleComplex
__half                      → __half  (不变, 来自 musa_fp16.h)

# ===== 枚举替换 =====
CUBLAS_OP_N / T / C         → MUBLAS_OP_N / T / C
CUBLAS_FILL_MODE_LOWER/UPPER→ MUBLAS_FILL_MODE_LOWER/UPPER (多 FULL=123 option)
CUBLAS_DIAG_NON_UNIT/UNIT   → MUBLAS_DIAG_NON_UNIT/UNIT
CUBLAS_SIDE_LEFT/RIGHT      → MUBLAS_SIDE_LEFT/RIGHT (多 BOTH=143 option)
CUBLAS_STATUS_SUCCESS       → MUBLAS_STATUS_SUCCESS
CUBLAS_STATUS_*             → MUBLAS_STATUS_*

# ===== 辅助函数 =====
cublasCreate                → mublasCreate
cublasDestroy               → mublasDestroy
cublasSetStream             → mublasSetStream
cublasGetStream             → mublasGetStream
cublasSetMathMode           → mublasSetMathMode
cublasGetMathMode           → mublasGetMathMode
cublasGetVersion            → mublasGetVersion
cublasSetPointerMode        → mublasSetPointerMode
cublasGetPointerMode        → mublasGetPointerMode
cublasSetAtomicsMode        → mublasSetAtomicsMode
cublasGetAtomicsMode        → mublasGetAtomicsMode
cublasSetWorkspace          → mublasSetWorkspace
cublasSetVectorAsync        → mublasSetVectorAsync
cublasGetVectorAsync        → mublasGetVectorAsync
cublasSetMatrixAsync        → mublasSetMatrixAsync
cublasGetMatrixAsync        → mublasGetMatrixAsync
```

## Python import 替换示例

```python
# 原 cuBLAS
from .cublasCcopy_v2 import cublasCcopy_v2
from .cublasSgemmEx import cublasSgemmEx
from .cublasSgemmBatched_64 import cublasSgemmBatched_64

# 改为 MUBLAS
from .mublasCcopy import mublasCcopy
from .mublasGemmEx import mublasGemmEx         # 注意: 去 S 前缀
from .mublasSgemmBatched import mublasSgemmBatched  # 去 _64
```

## 3 个例外

| # | cuBLAS | MUBLAS | 原因 |
|---|--------|--------|------|
| 1 | `cublasSgemmEx` | `mublasGemmEx` | 摩尔用泛型 `GemmEx` 无类型前缀 |
| 2 | `*_64` 系列 | 去掉 `_64` | 摩尔不存在 `_64` 变种，参数均为 `int` |
| 3 | `*_v2` 系列 | 去掉 `_v2` | 摩尔使用 C99 接口无版本后缀 |

## MUBLAS 版本信息

- 版本: **1.10.6**
- 库文件: `libmublas.so`, `libmublasLt.so`
- 头文件: `mublas.h`, `mublas_v2.h`(5.1.0), `mublasLt.h`, `mublasXt.h`
- 路径: `/usr/local/musa/include/`