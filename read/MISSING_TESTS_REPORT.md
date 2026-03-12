# 缺失测试的 API 报告

## 检查结果

### ✅ 总体情况

- **注册的算子总数**: 229
- **测试 label 总数**: 233
- **已映射的算子数**: 219
- **缺失测试的算子数**: 11

### ❌ 真正缺失测试的标准算子（6 个）

这些是标准 PyTorch 算子，应该有测试但当前没有：

1. **div** - 有测试 label `@label("div")`，但映射可能有问题
2. **div_** - 有测试 label `@label("div_")`，但映射可能有问题
3. **dropout** - 别名，有测试 label `@label("dropout")`，应该映射到 `native_dropout`
4. **group_norm** - 别名，有测试 label `@label("group_norm")`，应该映射到 `native_group_norm`
5. **layer_norm** - 别名，有测试 label `@label("layer_norm")`，应该映射到 `native_layer_norm`
6. **unique** - 有测试 label `@label("unique")`，应该映射到 `_unique2`

**注意**: 这些算子实际上**有测试**，但映射关系可能有问题，导致检查脚本认为它们没有测试。

### ⚠️ 其他缺失的算子（5 个）

这些可能是内部算子或自定义算子，可能不需要单独测试：

1. **_weight_norm** - PyTorch 内部算子
2. **constant_pad_nd** - PyTorch 内部算子
3. **diagonal_backward** - PyTorch 内部算子（backward kernel）
4. **gather_backward** - PyTorch 内部算子（backward kernel）
5. **weight_norm** - 自定义算子

## 问题分析

### 映射关系问题

检查脚本的映射逻辑可能有问题：

1. **div/divide**: `div` 和 `divide` 是不同的算子，都在 `IMPL_INFO` 中
   - 测试中有 `@label("div")`，应该直接匹配 `div`
   - 但检查脚本可能错误地认为需要映射

2. **别名处理**: `dropout`, `group_norm`, `layer_norm` 是别名
   - 它们在 `IMPL_INFO` 中已存在（通过 `IMPL_INFO["dropout"] = IMPL_INFO["native_dropout"]`）
   - 测试中有对应的 label
   - 但检查脚本可能没有正确处理

3. **unique**: 测试中有 `@label("unique")`，但注册的是 `_unique2`
   - 需要映射 `unique` -> `_unique2`
   - 映射表中有这个映射，但可能没有生效

## 建议

1. **修复映射逻辑**: 确保检查脚本正确处理别名和映射关系
2. **验证测试**: 确认这些算子是否真的有测试（通过实际运行测试）
3. **添加缺失测试**: 如果确实没有测试，建议添加

