# 测试 Label 到注册的反向检查报告

## 检查目的

确保所有测试中的 `@label("算子")` 标记都能找到对应的 flagbench API 注册（IMPL_INFO），这样测试时才会使用 Triton kernel，而不是 fallback 到 PyTorch kernel。

## 检查结果

### ✅ 总体情况

- **测试文件中的 label 总数**: 233 (排除 skip/skipif)
- **找到对应注册的 label**: 227
- **未找到对应注册的 label**: 4

### ❌ 未找到对应注册的测试 label (4 个)

#### 1. `avg_pool2d`
- **位置**: `test_special_ops.py`
- **状态**: 测试函数被注释掉了（`# @label("avg_pool2d")`）
- **原因**: 测试代码被注释，不需要注册
- **建议**: 如果将来启用这个测试，需要在 `IMPL_INFO` 中添加 `avg_pool2d` 的注册

#### 2. `flash_mla`
- **位置**: `test_attention_ops.py`
- **状态**: 测试函数存在，但未在 `IMPL_INFO` 中注册
- **原因**: 这是自定义算子，可能还没有实现 Triton kernel
- **建议**: 
  - 如果已有实现，需要在 `IMPL_INFO` 中添加 `flash_mla` 的注册
  - 如果没有实现，测试会 fallback 到 PyTorch kernel（如果存在）或报错

#### 3. `linear`
- **位置**: `test_blas_ops.py`
- **状态**: 测试函数存在，但未在 `IMPL_INFO` 中注册
- **原因**: `linear` 是 `addmm` 的别名，测试中同时使用了 `@label("linear")` 和 `@label("matmul")` 来标记 `addmm` 测试
- **建议**: 
  - 如果 `addmm` 已注册，可以考虑在映射表中添加 `linear` -> `addmm` 的映射
  - 或者移除 `@label("linear")` 和 `@label("matmul")`，只保留 `@label("addmm")`

#### 4. `matmul`
- **位置**: `test_blas_ops.py`
- **状态**: 测试函数存在，但未在 `IMPL_INFO` 中注册
- **原因**: `matmul` 是 `addmm` 的别名，测试中同时使用了 `@label("linear")` 和 `@label("matmul")` 来标记 `addmm` 测试
- **建议**: 
  - 如果 `addmm` 已注册，可以考虑在映射表中添加 `matmul` -> `addmm` 的映射
  - 或者移除 `@label("linear")` 和 `@label("matmul")`，只保留 `@label("addmm")`

## 已注册的相关算子

以下算子在 `IMPL_INFO` 中已注册：
- ✅ `addmm` - 矩阵乘法（带偏置）
- ✅ `mm` - 矩阵乘法
- ✅ `bmm` - 批量矩阵乘法

## 建议

### 1. 对于 `avg_pool2d`
- 如果测试被注释掉了，可以忽略
- 如果将来启用，需要添加注册

### 2. 对于 `flash_mla`
- 检查是否有 Triton 实现
- 如果有，添加到 `IMPL_INFO`
- 如果没有，考虑移除测试或标记为 skip

### 3. 对于 `linear` 和 `matmul`
- 这两个 label 用于标记 `addmm` 测试
- 建议在 `check_missing_tests.py` 的映射表中添加：
  ```python
  "linear": "addmm",
  "matmul": "addmm",
  ```
- 或者修改测试文件，移除多余的 label

## 总结

✅ **227/231 个测试 label 都能找到对应的注册**（排除被注释的 `avg_pool2d`）

⚠️ **4 个 label 需要注意**：
- `avg_pool2d` - 测试被注释，可忽略
- `flash_mla` - 自定义算子，需要确认是否有实现
- `linear` 和 `matmul` - 是 `addmm` 的别名，建议添加映射或修改测试

