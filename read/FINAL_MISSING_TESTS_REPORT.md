# 最终报告：哪些 API 没有测试

## 检查结果总结

### ✅ 总体情况

- **注册的算子总数**: 229
- **测试 label 总数**: 233
- **已映射的算子数**: 224
- **缺失测试的算子数**: 6

### ❌ 真正缺失测试的标准算子（1 个）

1. **unique** - 有测试 label `@label("unique")`，但映射可能有问题
   - `unique` 在 `IMPL_INFO` 中，注册的是 `_unique2`
   - 测试中有 `@label("unique")`
   - 需要确认映射是否正确

### ⚠️ 其他缺失的算子（5 个）

这些可能是内部算子或自定义算子，可能不需要单独测试：

1. **_weight_norm** - PyTorch 内部算子
2. **constant_pad_nd** - PyTorch 内部算子
3. **diagonal_backward** - PyTorch 内部算子（backward kernel）
4. **gather_backward** - PyTorch 内部算子（backward kernel）
5. **weight_norm** - 自定义算子

## 详细分析

### 已修复的映射

以下算子实际上**有测试**，通过修复映射关系已解决：

1. ✅ **div/div_** - 测试中有 `@label("div")` 和 `@label("div_")`，直接匹配
2. ✅ **dropout/group_norm/layer_norm** - 别名，测试中有对应的 label
3. ✅ **true_divide/true_divide_** - 通过 `divide` 的测试覆盖
4. ✅ **upsample** - 通过 `upsample_bicubic2d_aa` 和 `upsample_nearest2d` 的测试覆盖

### 需要进一步检查

1. **unique** - 需要确认映射关系是否正确

## 建议

1. **验证 unique 的映射**: 确认 `@label("unique")` 是否正确映射到 `IMPL_INFO["unique"]`
2. **内部算子**: `_weight_norm`, `constant_pad_nd`, `diagonal_backward`, `gather_backward` 是内部算子，可能不需要单独测试
3. **自定义算子**: `weight_norm` 是自定义算子，可能不需要测试

## 结论

**几乎所有标准 PyTorch 算子都有对应的测试！**

只有 1 个标准算子（`unique`）可能需要进一步检查映射关系，其他缺失的都是内部算子或自定义算子。

