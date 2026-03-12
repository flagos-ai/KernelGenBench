# 内部算子说明

## 什么是内部算子？

**内部算子**（Internal Operators）是 PyTorch 内部使用的底层实现，不是用户直接调用的 API。

### 特点

1. **不在 `PYTORCH_OPERATORS` 中**：这些算子不是标准的 PyTorch 用户 API
2. **在 `IMPL_INFO` 中**：它们需要注册以实现自定义的 Triton kernel
3. **用户通过其他 API 调用**：用户通过高级 API 间接使用这些算子

## 示例

### 1. `constant_pad_nd`

- **用户 API**: `torch.nn.functional.pad`
- **内部实现**: `constant_pad_nd`
- **说明**: `pad` 函数内部会调用 `constant_pad_nd` 来实现填充操作

```python
# 用户代码
x = torch.nn.functional.pad(input, pad, mode='constant', value=0)

# 内部实现
# torch.nn.functional.pad -> constant_pad_nd
```

**是否需要在 `PYTORCH_OPERATORS` 中？**
- ❌ **不需要**：`constant_pad_nd` 不是用户 API
- ✅ **需要**：在 `IMPL_INFO` 中注册，以便用 Triton kernel 替换内部实现

### 2. `diagonal_backward`

- **用户 API**: `torch.diagonal`
- **内部实现**: `diagonal_backward`（backward kernel）
- **说明**: `diagonal` 的前向和反向传播分别有不同的实现

```python
# 用户代码
x = torch.diagonal(input, offset=0, dim1=0, dim2=1)

# 内部实现
# forward: diagonal
# backward: diagonal_backward
```

**是否需要在 `PYTORCH_OPERATORS` 中？**
- ❌ **不需要**：`diagonal_backward` 不是用户 API，是自动求导系统内部使用的
- ✅ **需要**：在 `IMPL_INFO` 中注册，以便用 Triton kernel 替换 backward 实现

### 3. `gather_backward`

- **用户 API**: `torch.gather`
- **内部实现**: `gather_backward`（backward kernel）
- **说明**: `gather` 的反向传播实现

**是否需要在 `PYTORCH_OPERATORS` 中？**
- ❌ **不需要**：`gather_backward` 不是用户 API
- ✅ **需要**：在 `IMPL_INFO` 中注册，以便用 Triton kernel 替换 backward 实现

## 总结

### 内部算子列表

以下算子是内部算子，**不需要**在 `PYTORCH_OPERATORS` 中：

1. `constant_pad_nd` - `torch.nn.functional.pad` 的内部实现
2. `diagonal_backward` - `torch.diagonal` 的 backward kernel
3. `gather_backward` - `torch.gather` 的 backward kernel
4. `_weight_norm` - `torch._weight_norm` 的内部实现（下划线开头表示内部）

### 规则

- ✅ **在 `IMPL_INFO` 中注册**：所有需要替换的算子（包括内部算子）
- ❌ **不在 `PYTORCH_OPERATORS` 中**：内部算子不是用户 API
- ✅ **在 `PYTORCH_OPERATORS` 中**：只有用户直接调用的标准 PyTorch API

### 如何判断？

1. **下划线开头**：通常表示内部实现（如 `_weight_norm`, `_unique2`）
2. **`_backward` 后缀**：backward kernel（如 `diagonal_backward`, `gather_backward`）
3. **不在 PyTorch 文档中**：用户无法直接调用的 API
4. **通过其他 API 间接使用**：如 `constant_pad_nd` 通过 `pad` 使用

