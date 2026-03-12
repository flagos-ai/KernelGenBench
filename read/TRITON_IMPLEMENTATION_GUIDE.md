# Kaldi cudamatrix Triton 实现指南

## 架构理解

### 层次结构

```
用户代码
  ↓
cu-matrix.h (类接口层) - CuMatrixBase::CopyCols(), AddRows() 等
  ↓
cu-math.h (函数接口层) - RegularizeL1(), Randomize(), Splice() 等
  ↓
cu-kernels.h (C++ wrapper层) - cuda_add_row_sum_mat(), cuda_randomize() 等
  ↓
cu-kernels.cu (底层kernel实现) - __global__ 函数
```

### 关键理解

**✅ 正确做法**: 从 **wrapper 层**（cu-math.h, cu-matrix.h）提取接口，用 Triton 实现整个 wrapper 函数的功能

**❌ 错误做法**: 直接看 cu-kernels.cu 中的底层 kernel，因为：
1. 一个 wrapper 函数可能调用多个底层 kernel
2. wrapper 层定义了用户实际使用的接口语义
3. Triton 应该替换整个功能单元，而不是单个 kernel

---

## 实现策略

### 1. 从 cu-math.h 提取函数接口

**文件**: `cu-math.h`

这些是独立的函数，可以直接用 Triton 替换：

- `RegularizeL1()` - L1正则化
- `Randomize()` - 随机化行
- `Splice()` - 拼接帧
- `Copy()` - 复制列
- `EnsureNonzero()` - 确保非零
- `ComputeLstmNonlinearity()` - LSTM非线性计算
- `BackpropLstmNonlinearity()` - LSTM反向传播
- `NormalizePerRow()` - 按行归一化
- `DiffNormalizePerRow()` - 按行归一化差分

**示例**:
```cpp
// cu-math.h 中的接口
template<typename Real>
void RegularizeL1(CuMatrixBase<Real> *weight, CuMatrixBase<Real> *gradient,
                  Real l1_penalty, Real learning_rate);

// cu-math.cc 中的实现调用底层 kernel
cuda_regularize_l1(dimGrid, dimBlock, weight->Data(), grad->Data(), ...);

// 用 Triton 实现时，应该实现整个 RegularizeL1 的功能
```

### 2. 从 cu-matrix.h 提取类成员函数

**文件**: `cu-matrix.h`

这些是 CuMatrixBase 类的成员函数，需要实现：

- `CopyCols()` - 复制列
- `AddCols()` - 添加列
- `CopyRows()` - 复制行
- `AddRows()` - 添加行
- `MulRows()` - 行乘法
- `AddToRows()` - 添加到行
- `SumColumnRanges()` - 列范围求和
- `AddRowRanges()` - 行范围添加
- 等等...

**示例**:
```cpp
// cu-matrix.h 中的接口
void CopyCols(const CuMatrixBase<Real> &src,
              const CuArrayBase<MatrixIndexT> &indexes);

// 用 Triton 实现时，应该实现整个 CopyCols 的功能
```

### 3. 从 cu-kernels.h 了解底层调用

**文件**: `cu-kernels.h`

这些是 C++ inline wrapper，调用 ANSI-C CUDA kernel：

- `cuda_add_row_sum_mat()` - 行求和
- `cuda_add_col_sum_mat()` - 列求和
- `cuda_randomize()` - 随机化
- `cuda_splice()` - 拼接
- 等等...

**注意**: 这些函数通常有 float 和 double 两个版本（cudaF_xxx 和 cudaD_xxx）

---

## 提取算子列表的方法

### 步骤 1: 提取 cu-math.h 中的函数

```bash
# 提取所有模板函数声明
grep -E "^template|^void" cu-math.h | grep -v "^//"
```

### 步骤 2: 提取 cu-matrix.h 中的成员函数

```bash
# 提取类成员函数
grep -E "^\s+(void|Real|bool|int)" cu-matrix.h | grep "("
```

### 步骤 3: 查看 cu-math.cc 了解实现逻辑

查看每个函数在 `cu-math.cc` 中的实现，了解：
- 调用了哪些底层 kernel
- 参数如何转换
- 是否有 CPU fallback 逻辑

---

## 实现优先级建议

### 高优先级（常用且相对简单）

1. **cu-math.h**:
   - `RegularizeL1()` - L1正则化
   - `Randomize()` - 随机化
   - `Splice()` - 拼接
   - `Copy()` - 复制列
   - `EnsureNonzero()` - 确保非零

2. **cu-matrix.h**:
   - `CopyCols()` - 复制列
   - `CopyRows()` - 复制行
   - `AddRows()` - 添加行
   - `AddCols()` - 添加列

### 中优先级（常用但复杂）

3. **cu-math.h**:
   - `NormalizePerRow()` - 按行归一化
   - `ComputeLstmNonlinearity()` - LSTM计算
   - `BackpropLstmNonlinearity()` - LSTM反向传播

4. **cu-matrix.h**:
   - `SumColumnRanges()` - 列范围求和
   - `AddRowRanges()` - 行范围添加
   - `MulRows()` - 行乘法

### 低优先级（特殊用途）

5. 其他专用函数

---

## 实现示例

### 示例 1: RegularizeL1

**原始实现** (cu-math.cc):
```cpp
template<typename Real>
void RegularizeL1(CuMatrixBase<Real> *weight, CuMatrixBase<Real> *grad, 
                  Real l1, Real lr) {
  dim3 dimBlock(CU2DBLOCK, CU2DBLOCK);
  dim3 dimGrid(n_blocks(weight->NumCols(), CU2DBLOCK), 
               n_blocks(weight->NumRows(), CU2DBLOCK));
  cuda_regularize_l1(dimGrid, dimBlock, weight->Data(), grad->Data(), 
                     l1, lr, weight->Dim(), grad->Stride());
}
```

**Triton 实现思路**:
- 实现整个 `RegularizeL1` 函数的功能
- 处理 L1 正则化的逻辑（包括跨零处理）
- 不需要关心底层 kernel 的细节

### 示例 2: CopyCols

**原始实现** (cu-matrix.cc):
```cpp
void CopyCols(const CuMatrixBase<Real> &src,
              const CuArrayBase<MatrixIndexT> &indexes) {
  // 调用 cuda_copy_cols kernel
}
```

**Triton 实现思路**:
- 实现整个 `CopyCols` 的功能
- 处理索引映射和边界情况（-1 索引）
- 一次性完成所有列的复制

---

## 注意事项

1. **参数类型**: 注意 float 和 double 两个版本
2. **边界处理**: 注意 -1 索引等特殊值
3. **内存布局**: 注意 stride 和连续内存的区别
4. **CPU fallback**: 了解 CPU 版本的逻辑，确保 Triton 版本行为一致

---

## 总结

✅ **应该关注**: cu-math.h 和 cu-matrix.h 中的函数接口
✅ **应该实现**: 整个 wrapper 函数的功能
❌ **不需要**: 直接看 cu-kernels.cu 中的底层实现细节

**核心原则**: Triton 实现应该替换的是**用户可见的接口层**，而不是底层的 kernel 实现。

