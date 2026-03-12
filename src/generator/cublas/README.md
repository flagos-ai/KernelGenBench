# cuBLAS Baseline 函数生成记录

## 概述

本次工作为 10 个 cuBLAS 函数创建了 ctypes baseline 实现和对应的测试函数。

## 完成的函数列表

### BLAS Level 1
1. **cublasSaxpy_v2** - float32 向量加法：y = alpha * x + y

### BLAS Level 3 - GEMM (矩阵乘法)
2. **cublasSgemmStridedBatched** - float32 批量矩阵乘法
3. **cublasDgemmStridedBatched** - float64 批量矩阵乘法
4. **cublasHgemmStridedBatched** - float16 批量矩阵乘法
5. **cublasCgemmStridedBatched** - complex64 批量矩阵乘法
6. **cublasZgemmStridedBatched** - complex128 批量矩阵乘法

### BLAS Level 2 - GEMV (矩阵向量乘法)
7. **cublasSgemvStridedBatched** - float32 批量矩阵向量乘法
8. **cublasDgemvStridedBatched** - float64 批量矩阵向量乘法
9. **cublasCgemvStridedBatched** - complex64 批量矩阵向量乘法
10. **cublasZgemvStridedBatched** - complex128 批量矩阵向量乘法

## 文件位置

### Baseline 实现
- 路径：`/share/project/zpy/flagbench/src/flagbench/dataset/baseline/cublas_ctypes/`
- 实现方式：使用 Python ctypes 直接调用 cuBLAS C API
- 注册方式：通过 `__init__.py` 导出到 `flagbench.baseline` 命名空间

### 测试函数
- 路径：`/share/project/zpy/flagbench/src/generator/test_func/`
- 命名规则：`test_accuracy_{function_name}.py`
- 导入方式：直接导入 baseline 函数（按需加载）

## 测试结果

所有 10 个函数的测试均已通过：

| 函数名 | 测试用例数 | 通过率 | 测试时间 |
|--------|-----------|--------|---------|
| cublasSaxpy_v2 | 36 | 100% | ~2分钟 |
| cublasSgemmStridedBatched | 24 | 100% | ~16秒 |
| cublasDgemmStridedBatched | 16 | 100% | ~17秒 |
| cublasHgemmStridedBatched | 16 | 100% | ~15秒 |
| cublasCgemmStridedBatched | 16 | 100% | ~15秒 |
| cublasZgemmStridedBatched | 16 | 100% | ~16秒 |
| cublasSgemvStridedBatched | 16 | 100% | ~14秒 |
| cublasDgemvStridedBatched | 16 | 100% | ~16秒 |
| cublasCgemvStridedBatched | 16 | 100% | ~15秒 |
| cublasZgemvStridedBatched | 16 | 100% | ~15秒 |

**总计：180 个测试用例全部通过**

## 关键技术点

### 1. Transpose 标志转换
cuBLAS C API 使用整数表示转置操作：
```python
CUBLAS_OP_N = 0  # 不转置
CUBLAS_OP_T = 1  # 转置
trans_op = CUBLAS_OP_T if trans == 'T' else CUBLAS_OP_N
```

### 2. 直接导入方式
测试函数使用直接导入而非注册方式，避免全量加载：
```python
from flagbench.dataset.baseline.cublas_ctypes.cublasSaxpy_v2 import cublasSaxpy_v2 as baseline_cublasSaxpy_v2
```

### 3. Triton 命名空间
在 `flagbench/__init__.py` 中将 baseline 函数复制到 triton 命名空间：
```python
for attr_name in dir(baseline):
    if not attr_name.startswith('_'):
        setattr(triton, attr_name, getattr(baseline, attr_name))
```

### 4. 精度验证
使用 `assert_close` 函数进行精度验证，支持 `reduce_dim` 参数：
```python
# GEMM: 累加 K 次
assert_close(act_out, ref_out, dtype, reduce_dim=K)

# GEMV: 累加 N 或 M 次
assert_close(act_out, ref_out, dtype, reduce_dim=N if trans == 'N' else M)
```

## 测试命令

使用 `DISPATCH_TORCH_LIB=0` 模式测试 baseline 函数：

```bash
cd /share/project/zpy/flagbench
source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench

# 测试单个函数
python test/test_accuracy_ut.py \
    --name cublasSaxpy_v2 \
    --test-file generator.test_func.test_cublasSaxpy_v2 \
    --device-count 1 \
    --timeout 300
```

## 生成工具

使用金山云 API (gpt-5 模型) 生成 baseline 和测试函数：
- API 地址：`https://kspmas.ksyun.com/v1/chat/completions`
- 生成脚本：`/share/project/zpy/flagbench/src/generator/baseline/output/generate_remaining_ut.py`

## 注意事项

1. **cuBLAS 库路径**：所有 baseline 函数使用 `/usr/local/cuda/lib64/libcublas.so.12`
2. **延迟加载**：建议使用延迟加载避免导入时卡顿（已在 cublasSaxpy_v2 中实现）
3. **精度设置**：
   - float32: rtol=1e-2, atol=1e-2
   - float64: rtol=1e-5, atol=1e-5
   - 使用 `reduce_dim` 参数根据累加次数调整容差

## 完成时间

2026-02-04

## 参与人员

- 用户：zpy
- AI 助手：Claude Sonnet 4.5
