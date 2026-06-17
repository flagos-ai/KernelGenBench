# cuBLAS → MUBLAS 3 个问题具体修改方案

---

## 问题 1: `cublasSgemmEx` → `mublasGemmEx`

### 问题描述
函数名去掉了类型前缀 `S`，且 computeType/algo 枚举完全换了一套。

### 函数签名对照

```cpp
// cuBLAS 原版
cublasStatus_t cublasSgemmEx(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const void *alpha,
    const void *A, cudaDataType_t Atype, int lda,
    const void *B, cudaDataType_t Btype, int ldb,
    const void *beta,
    void *C, cudaDataType_t Ctype, int ldc,
    cudaDataType_t computeType,
    cublasGemmAlgo_t algo
);

// MUBLAS 新版
mublasStatus mublasGemmEx(
    mublasHandle_t handle,
    mublasOperation_t transA, mublasOperation_t transB,
    int m, int n, int k,
    const void *alpha,
    const void *a, musaDataType_t a_type, int lda,
    const void *b, musaDataType_t b_type, int ldb,
    const void *beta,
    void *c, musaDataType_t c_type, int ldc,
    mublasComputeType_t compute_type,   // ← 类型变了！
    mublasGemmAlgo_t algo               // ← 枚举变了！
);
```

### computeType 对照

| 用法场景 | cuBLAS `cudaDataType_t` | MUBLAS `mublasComputeType_t` |
|----------|------------------------|------------------------------|
| FP16 输入, FP16 累加 | `CUDA_R_16F` (值=2) | `MUBLAS_COMPUTE_16F` (值=64) |
| FP16 输入, FP32 累加 | `CUDA_R_32F` (值=0) | `MUBLAS_COMPUTE_32F` (值=68) |
| FP32 输入, FP32 累加 | `CUDA_R_32F` (值=0) | `MUBLAS_COMPUTE_32F` (值=68) |
| TF32 输入, FP32 累加 | `CUDA_R_32F_FAST_TF32` (值=...) | `MUBLAS_COMPUTE_32F_FAST_TF32` (值=77) |
| FP64 输入, FP64 累加 | `CUDA_R_64F` (值=4) | `MUBLAS_COMPUTE_64F` (值=70) |
| INT32 输入, INT32 累加 | `CUDA_R_32I` (值=8) | `MUBLAS_COMPUTE_32I` (值=72) |

**关键**: 不能只替换宏名，因为数值完全不同！

### algo 对照

| cuBLAS `cublasGemmAlgo_t` | MUBLAS `mublasGemmAlgo_t` |
|---------------------------|---------------------------|
| `CUBLAS_GEMM_DEFAULT` (很多值可用) | `MUBLAS_GEMM_DEFAULT` (值=0x0) |
| `CUBLAS_GEMM_DEFAULT_TENSOR_OP` | `MUBLAS_GEMM_DEFAULT_TENSOR_OP` (值=0x1) |

⚠️ cuBLAS 有 `CUBLAS_GEMM_ALGO0` 到 `CUBLAS_GEMM_ALGO15`、`CUBLAS_GEMM_DFALT` 等多个 algo，
MUBLAS 只有 2 个。不能用数值索引。

### 迁移代码 (C++ 封装方案)

如果你有很多地方调用 `cublasSgemmEx`，最干净的方案是写一个兼容宏或内联函数：

```cpp
// === 方案 A: 内联兼容函数 (推荐) ===
inline mublasStatus mublasSgemmEx(
    mublasHandle_t handle,
    mublasOperation_t transA, mublasOperation_t transB,
    int m, int n, int k,
    const void *alpha,
    const void *A, musaDataType_t Atype, int lda,
    const void *B, musaDataType_t Btype, int ldb,
    const void *beta,
    void *C, musaDataType_t Ctype, int ldc,
    musaDataType_t oldComputeType,  // 旧的 cudaDataType_t 值
    mublasGemmAlgo_t algo)
{
    // 把旧 cudaDataType_t 映射到 mublasComputeType_t
    mublasComputeType_t newComputeType;
    switch (oldComputeType) {
        case 2:  // CUDA_R_16F 或 musa 16F
            newComputeType = MUBLAS_COMPUTE_16F;   // 64
            break;
        case 0:  // CUDA_R_32F 或 musa 32F
            newComputeType = MUBLAS_COMPUTE_32F;   // 68
            break;
        case 4:  // CUDA_R_64F 或 musa 64F
            newComputeType = MUBLAS_COMPUTE_64F;   // 70
            break;
        case 8:  // CUDA_R_32I 或 musa 32I
            newComputeType = MUBLAS_COMPUTE_32I;   // 72
            break;
        default:
            newComputeType = MUBLAS_COMPUTE_32F;   // fallback
    }
    return mublasGemmEx(handle, transA, transB, m, n, k,
                        alpha, A, Atype, lda, B, Btype, ldb,
                        beta, C, Ctype, ldc,
                        newComputeType, algo);
}

// 然后全局替换:
// cublasSgemmEx(  →  mublasSgemmEx(
```
```cpp
// === 方案 B: 逐处替换 ===
// 旧代码:
cublasSgemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              M, N, K,
              &alpha, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb,
              &beta, C, CUDA_R_16F, ldc,
              CUDA_R_32F, CUBLAS_GEMM_DEFAULT);

// 新代码:
mublasGemmEx(handle, MUBLAS_OP_N, MUBLAS_OP_N,
             M, N, K,
             &alpha, A, MUSA_R_16F, lda, B, MUSA_R_16F, ldb,
             &beta, C, MUSA_R_16F, ldc,
             MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
```
```cpp
// === 方案 C: Python/ctypes 用法 ===
// 如果通过 ctypes 调用，直接改函数名和参数值:

// 旧:
lib.cublasSgemmEx(handle, 0, 0,  # CUBLAS_OP_N = 0,0
#                  m, n, k,
#                  alpha_ptr, a_ptr, 2, lda,  # CUDA_R_16F = 2
#                  b_ptr, 2, ldb,
#                  beta_ptr, c_ptr, 2, ldc,
#                  0, 0)  # CUDA_R_32F = 0, CUBLAS_GEMM_DEFAULT = 0

// 新:
lib.mublasGemmEx(handle, 111, 111,  # MUBLAS_OP_N = 111,111
#                 m, n, k,
#                 alpha_ptr, a_ptr, 2, lda,   # musa 16F API 值也是 2
#                 b_ptr, 2, ldb,
#                 beta_ptr, c_ptr, 2, ldc,
#                 68, 0)  # MUBLAS_COMPUTE_32F = 68, MUBLAS_GEMM_DEFAULT = 0
```

---

## 问题 2: `_64` 后缀 → 参数类型 int64_t → int

### 问题描述
cuBLAS 中带 `_64` 后缀的函数使用 `int64_t` 作为矩阵维度参数。
MUBLAS 去掉了 `_64` 变种，所有维度都用 `int` (int32_t)。

### 受影响函数 (4个)

| cuBLAS | MUBLAS |
|--------|--------|
| `cublasCgemmStridedBatched_64` | `mublasCgemmStridedBatched` |
| `cublasDgemmStridedBatched_64` | `mublasDgemmStridedBatched` |
| `cublasSgemmBatched_64` | `mublasSgemmBatched` |
| `cublasCgemvBatched_64` | `mublasCgemvBatched` |

### 迁移代码

```cpp
// === 旧代码 ===
int64_t M = 65536, N = 65536, K = 65536;
cublasDgemmStridedBatched_64(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              M, N, K,          // int64_t
                              &alpha, A, lda, strideA,
                              B, ldb, strideB,
                              &beta, C, ldc, strideC, batchCount);

// === 新代码 ===
int M = 65536, N = 65536, K = 65536;  // int64_t → int
mublasDgemmStridedBatched(handle, MUBLAS_OP_N, MUBLAS_OP_N,
                          M, N, K,              // int
                          &alpha, A, lda, strideA,
                          B, ldb, strideB,
                          &beta, C, ldc, strideC, batchCount);
```

```cpp
// === 如果一定有超大矩阵 (M*N*K > 2^31-1) ===
// 方案 1: 拆成多个子矩阵分次计算
// 方案 2: 检查 MUBLAS 新版是否重新提供了 _64 变种
//         (MUBLAS 1.10.6 目前没有，未来可能加)
// 方案 3: 使用 GemmEx + 手动分块

// 安全检查宏:
#define MUBLAS_CHECK_SIZE(val, name) \
    do { \
        if ((val) > INT32_MAX) { \
            fprintf(stderr, "%s exceeds INT32_MAX: %ld\n", name, (long)(val)); \
            return MUBLAS_STATUS_INVALID_SIZE; \
        } \
    } while(0)
```

### stride 参数
stride 参数（`strideA`, `strideB`, `strideC`）在 MUBLAS 中仍然是 `long long int` (64位)，**不受影响**。只有 m, n, k 维度收窄为 int。

---

## 问题 3: `cudaStream_t` → `MUstream`

### 问题描述
MUBLAS 的 stream 类型是 `MUstream` 而非 `cudaStream_t`。

但两者底层是兼容的，都指向 `struct MUstream_st*`。

### 迁移代码

```cpp
// === 旧代码 ===
cudaStream_t stream;
cudaStreamCreate(&stream);
cublasSetStream(handle, stream);
cublasSetVectorAsync(n, elemSize, x, incx, y, incy, stream);

// === 新代码 ===
MUstream stream;                        // cudaStream_t → MUstream
muStreamCreate(&stream);                // cudaStreamCreate → muStreamCreate
mublasSetStream(handle, stream);
mublasSetVectorAsync(n, elemSize, x, incx, y, incy, stream);

// === 如果不想改接口，用 typedef 桥接 ===
// 方案: 在公共头文件中加入:
#if defined(__MUSACC__)
    typedef MUstream cudaStream_t;      // MUSA 上 cudaStream_t 就是 MUstream
#endif
// 这样 mublasSetStream(handle, cudaStream_t_var) 直接可用
```

```cpp
// === 所有受影响的 mublas 辅助函数 ===
// mublasSetStream(handle, MUstream)
// mublasGetStream(handle, MUstream*)
// mublasSetKernelStream(handle, MUstream)       // MUBLAS 新增
// mublasSetVectorAsync(..., MUstream)
// mublasGetVectorAsync(..., MUstream)
// mublasSetMatrixAsync(..., MUstream)
// mublasGetMatrixAsync(..., MUstream)
```

---

## 附: 枚举值问题 —— 推荐修复方案

**这是 50 个函数都受影响的问题**，不只是那 3 个。

### 方案 A: 统一头文件桥接 (推荐)

```c
// mublas_compat.h — 包含此头文件即可安全迁移
#include <mublas.h>

// 如果代码里还有裸数字，加编译期检查
#if defined(__MUSACC__)
    // 强制使用命名常量
    #define MUBLAS_SAFE_OP_N    MUBLAS_OP_N
    #define MUBLAS_SAFE_OP_T    MUBLAS_OP_T
    #define MUBLAS_SAFE_FILL_L  MUBLAS_FILL_MODE_LOWER
    #define MUBLAS_SAFE_FILL_U  MUBLAS_FILL_MODE_UPPER
    // ... 等等
#endif
```

### 方案 B: 逐文件 sed 检查

```bash
# 检查是否有裸数字用作枚举 (只检查特定场景)
# 匹配带 trans/op/fill/diag/side 参数位置的可能裸数字
grep -rnP "(trans|op|fill|diag|side).*,\s*[012]\s*,.*\)" *.c *.cpp *.cu

# 如果搜到结果，该处可能传了裸数字而不是枚举常量，需要改为:
#   0 → MUBLAS_OP_N / MUBLAS_FILL_MODE_LOWER / MUBLAS_DIAG_NON_UNIT / MUBLAS_SIDE_LEFT
#   1 → MUBLAS_OP_T / MUBLAS_FILL_MODE_UPPER / MUBLAS_DIAG_UNIT / MUBLAS_SIDE_RIGHT
#   2 → MUBLAS_OP_C
```

### 方案 C: Python 侧迁移

```python
# 旧代码
class CublasOp:
    N = 0
    T = 1
    C = 2

class CublasFill:
    LOWER = 0
    UPPER = 1

# 新代码
class MublasOp:
    N = 111
    T = 112
    C = 113

class MublasFill:
    UPPER = 121
    LOWER = 122
    FULL  = 123    # MUBLAS 新增

class MublasDiag:
    NON_UNIT = 131
    UNIT     = 132

class MublasSide:
    LEFT  = 141
    RIGHT = 142
    BOTH  = 143    # MUBLAS 新增
```

---

## 完整迁移脚本

```bash
#!/bin/bash
# migrate_cublas_to_mublas.sh — 自动迁移脚本

TARGET_DIR=${1:-.}

echo "=== Step 1: 函数名替换 ==="
# 基本替换
find "$TARGET_DIR" -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.py" | while read f; do
    # _v2 后缀去掉
    sed -i 's/cublasCcopy_v2/mublasCcopy/g' "$f"
    sed -i 's/cublasCdotu_v2/mublasCdotu/g' "$f"
    sed -i 's/cublasZdotc_v2/mublasZdotc/g' "$f"
    sed -i 's/cublasCgemm_v2/mublasCgemm/g' "$f"
    sed -i 's/cublasCgemv_v2/mublasCgemv/g' "$f"
    sed -i 's/cublasCgeru_v2/mublasCgeru/g' "$f"
    sed -i 's/cublasCsymm_v2/mublasCsymm/g' "$f"
    sed -i 's/cublasCsymv_v2/mublasCsymv/g' "$f"
    sed -i 's/cublasDasum_v2/mublasDasum/g' "$f"
    sed -i 's/cublasDaxpy_v2/mublasDaxpy/g' "$f"
    sed -i 's/cublasDcopy_v2/mublasDcopy/g' "$f"
    sed -i 's/cublasDgemm_v2/mublasDgemm/g' "$f"
    sed -i 's/cublasDgemv_v2/mublasDgemv/g' "$f"
    sed -i 's/cublasDsbmv_v2/mublasDsbmv/g' "$f"
    sed -i 's/cublasDsyr2_v2/mublasDsyr2/g' "$f"
    sed -i 's/cublasSaxpy_v2/mublasSaxpy/g' "$f"
    sed -i 's/cublasSdot_v2/mublasSdot/g' "$f"
    sed -i 's/cublasSgemm_v2/mublasSgemm/g' "$f"
    sed -i 's/cublasSger_v2/mublasSger/g' "$f"
    sed -i 's/cublasSscal_v2/mublasSscal/g' "$f"
    sed -i 's/cublasSsyrk_v2/mublasSsyrk/g' "$f"
    sed -i 's/cublasStbmv_v2/mublasStbmv/g' "$f"
    sed -i 's/cublasStrsm_v2/mublasStrsm/g' "$f"
    sed -i 's/cublasStrsv_v2/mublasStrsv/g' "$f"
    sed -i 's/cublasZgerc_v2/mublasZgerc/g' "$f"
    sed -i 's/cublasZswap_v2/mublasZswap/g' "$f"

    # 不带 _v2 的
    sed -i 's/cublasCgemmStridedBatched\b/mublasCgemmStridedBatched/g' "$f"
    sed -i 's/cublasDgemmStridedBatched\b/mublasDgemmStridedBatched/g' "$f"
    sed -i 's/cublasHgemmStridedBatched\b/mublasHgemmStridedBatched/g' "$f"
    sed -i 's/cublasSgemmStridedBatched\b/mublasSgemmStridedBatched/g' "$f"
    sed -i 's/cublasCgemvStridedBatched/mublasCgemvStridedBatched/g' "$f"
    sed -i 's/cublasDgemvStridedBatched/mublasDgemvStridedBatched/g' "$f"
    sed -i 's/cublasSgemvStridedBatched/mublasSgemvStridedBatched/g' "$f"
    sed -i 's/cublasZgemvStridedBatched/mublasZgemvStridedBatched/g' "$f"
    sed -i 's/cublasCgemmStridedBatched/mublasCgemmStridedBatched/g' "$f"
    sed -i 's/cublasDgemmBatched/mublasDgemmBatched/g' "$f"
    sed -i 's/cublasHgemmBatched/mublasHgemmBatched/g' "$f"
    sed -i 's/cublasZgemmBatched/mublasZgemmBatched/g' "$f"
    sed -i 's/cublasDgemvBatched/mublasDgemvBatched/g' "$f"
    sed -i 's/cublasSgemvBatched/mublasSgemvBatched/g' "$f"
    sed -i 's/cublasCgemvBatched/mublasCgemvBatched/g' "$f"
    sed -i 's/cublasZgemvBatched/mublasZgemvBatched/g' "$f"
    sed -i 's/cublasDtrsmBatched/mublasDtrsmBatched/g' "$f"
    sed -i 's/cublasZtrsmBatched/mublasZtrsmBatched/g' "$f"
    sed -i 's/cublasSdgmm/mublasSdgmm/g' "$f"
    sed -i 's/cublasSgeam/mublasSgeam/g' "$f"
    sed -i 's/cublasCsyrkEx/mublasCsyrkEx/g' "$f"
done

echo "=== Step 2: _64 后缀去掉 ==="
find "$TARGET_DIR" -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.py" | while read f; do
    sed -i 's/cublasCgemmStridedBatched_64/mublasCgemmStridedBatched/g' "$f"
    sed -i 's/cublasDgemmStridedBatched_64/mublasDgemmStridedBatched/g' "$f"
    sed -i 's/cublasSgemmBatched_64/mublasSgemmBatched/g' "$f"
    sed -i 's/cublasCgemvBatched_64/mublasCgemvBatched/g' "$f"
done

echo "=== Step 3: cublasSgemmEx → mublasGemmEx (特殊处理) ==="
find "$TARGET_DIR" -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.py" | while read f; do
    sed -i 's/cublasSgemmEx\b/mublasGemmEx/g' "$f"
done

echo "=== Step 4: 辅助函数 ==="
find "$TARGET_DIR" -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.py" | while read f; do
    sed -i 's/cublasCreate/mublasCreate/g' "$f"
    sed -i 's/cublasDestroy/mublasDestroy/g' "$f"
    sed -i 's/cublasSetStream/mublasSetStream/g' "$f"
    sed -i 's/cublasGetStream/mublasGetStream/g' "$f"
    sed -i 's/cublasSetMathMode/mublasSetMathMode/g' "$f"
    sed -i 's/cublasGetMathMode/mublasGetMathMode/g' "$f"
    sed -i 's/cublasGetVersion/mublasGetVersion/g' "$f"
    sed -i 's/cublasSetPointerMode/mublasSetPointerMode/g' "$f"
    sed -i 's/cublasGetPointerMode/mublasGetPointerMode/g' "$f"
    sed -i 's/cublasSetAtomicsMode/mublasSetAtomicsMode/g' "$f"
    sed -i 's/cublasGetAtomicsMode/mublasGetAtomicsMode/g' "$f"
    sed -i 's/cublasSetWorkspace/mublasSetWorkspace/g' "$f"
    sed -i 's/cublasSetVectorAsync/mublasSetVectorAsync/g' "$f"
    sed -i 's/cublasGetVectorAsync/mublasGetVectorAsync/g' "$f"
    sed -i 's/cublasSetMatrixAsync/mublasSetMatrixAsync/g' "$f"
    sed -i 's/cublasGetMatrixAsync/mublasGetMatrixAsync/g' "$f"
done

echo "=== Step 5: 枚举/类型替换 ==="
find "$TARGET_DIR" -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.py" | while read f; do
    sed -i 's/CUBLAS_OP_N\b/MUBLAS_OP_N/g' "$f"
    sed -i 's/CUBLAS_OP_T\b/MUBLAS_OP_T/g' "$f"
    sed -i 's/CUBLAS_OP_C\b/MUBLAS_OP_C/g' "$f"
    sed -i 's/CUBLAS_FILL_MODE_LOWER/MUBLAS_FILL_MODE_LOWER/g' "$f"
    sed -i 's/CUBLAS_FILL_MODE_UPPER/MUBLAS_FILL_MODE_UPPER/g' "$f"
    sed -i 's/CUBLAS_DIAG_NON_UNIT/MUBLAS_DIAG_NON_UNIT/g' "$f"
    sed -i 's/CUBLAS_DIAG_UNIT/MUBLAS_DIAG_UNIT/g' "$f"
    sed -i 's/CUBLAS_SIDE_LEFT\b/MUBLAS_SIDE_LEFT/g' "$f"
    sed -i 's/CUBLAS_SIDE_RIGHT\b/MUBLAS_SIDE_RIGHT/g' "$f"
    sed -i 's/CUBLAS_STATUS_SUCCESS/MUBLAS_STATUS_SUCCESS/g' "$f"
    sed -i 's/CUBLAS_POINTER_MODE_HOST/MUBLAS_POINTER_MODE_HOST/g' "$f"
    sed -i 's/CUBLAS_POINTER_MODE_DEVICE/MUBLAS_POINTER_MODE_DEVICE/g' "$f"
    sed -i 's/CUBLAS_GEMM_DEFAULT\b/MUBLAS_GEMM_DEFAULT/g' "$f"
    sed -i 's/CUBLAS_GEMM_DEFAULT_TENSOR_OP/MUBLAS_GEMM_DEFAULT_TENSOR_OP/g' "$f"
    # 类型
    sed -i 's/cublasHandle_t/mublasHandle_t/g' "$f"
    sed -i 's/cublasStatus_t/mublasStatus/g' "$f"
    sed -i 's/cublasOperation_t/mublasOperation_t/g' "$f"
    sed -i 's/cublasFillMode_t/mublasFillMode_t/g' "$f"
    sed -i 's/cublasDiagType_t/mublasDiagType_t/g' "$f"
    sed -i 's/cublasSideMode_t/mublasSideMode_t/g' "$f"
    sed -i 's/cuComplex\b/muComplex/g' "$f"
    sed -i 's/cuDoubleComplex/muDoubleComplex/g' "$f"
    sed -i 's/cudaStream_t/MUstream/g' "$f"
done

echo "=== Step 6: 头文件和链接 ==="
find "$TARGET_DIR" -name "*.c" -o -name "*.cpp" -o -name "*.h" -o -name "*.cu" -o -name "*.py" | while read f; do
    sed -i 's|#include <cublas_v2.h>|#include <mublas.h>|g' "$f"
    sed -i 's|#include <cuda_fp16.h>|#include <musa_fp16.h>|g' "$f"
    sed -i 's/-lcublas/-lmublas/g' "$f"
    # shared library
    sed -i 's/libcublas/libmublas/g' "$f"
done

echo "=== Done ==="
echo "⚠️  请手动检查:"
echo "  1. cublasSgemmEx → mublasGemmEx 的 computeType 参数值"
echo "  2. _64 函数的 int64_t→int 类型转换"
echo "  3. 是否有裸数字 0/1/2 用作枚举 (grep -rnP '(trans|op|fill|diag|side).*[012]' .)"
echo "  4. cudaStream_t 相关流操作是否需要适配 muStreamCreate/muStreamDestroy"
```