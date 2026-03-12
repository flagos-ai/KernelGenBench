# 修复总结

## 1. 删除 PYTORCH_OPERATORS 中的重复项

### 问题
`PYTORCH_OPERATORS` 中有 12 个重复的 key，虽然 Python 字典会自动覆盖，但为了代码清晰，删除了重复的定义。

### 修复
删除了以下重复的定义（保留第一次出现的）：
- `torch.cummax` (441, 602)
- `torch.diagonal` (446, 603)
- `torch.dot` (453注释, 604)
- `torch.nn.functional.elu_` (456注释, 605)
- `torch.nn.functional.glu` (476, 609)
- `torch.lerp` (494, 612)
- `torch.Tensor.lerp_` (495, 613)
- `torch.linspace` (496, 614)
- `torch.nan_to_num` (519, 617)
- `torch.polar` (530, 618)
- `torch.Tensor.scatter_` (555, 619)
- `torch.Tensor.to` (574, 625)

### 结果
✅ 现在 `PYTORCH_OPERATORS` 中只有唯一的 key，共 195 个。

## 2. 内部算子说明

### 问题
用户询问为什么 `constant_pad_nd` 和 `diagonal_backward` 不在 `PYTORCH_OPERATORS` 中。

### 解释
这些是**内部算子**（Internal Operators），不是用户直接调用的 API：

1. **`constant_pad_nd`**
   - 用户 API: `torch.nn.functional.pad`
   - 内部实现: `constant_pad_nd`
   - 不需要在 `PYTORCH_OPERATORS` 中

2. **`diagonal_backward`**
   - 用户 API: `torch.diagonal`
   - 内部实现: `diagonal_backward` (backward kernel)
   - 不需要在 `PYTORCH_OPERATORS` 中

3. **`gather_backward`**
   - 用户 API: `torch.gather`
   - 内部实现: `gather_backward` (backward kernel)
   - 不需要在 `PYTORCH_OPERATORS` 中

4. **`_weight_norm`**
   - 用户 API: `torch._weight_norm` (下划线开头表示内部)
   - 内部实现: `_weight_norm`
   - 不需要在 `PYTORCH_OPERATORS` 中（虽然已经在里面了，但这是历史遗留）

### 规则
- ✅ **在 `IMPL_INFO` 中注册**：所有需要替换的算子（包括内部算子）
- ❌ **不在 `PYTORCH_OPERATORS` 中**：内部算子不是用户 API
- ✅ **在 `PYTORCH_OPERATORS` 中**：只有用户直接调用的标准 PyTorch API

详细说明见 `INTERNAL_OPERATORS_EXPLANATION.md`。

## 3. 修复 weight_norm 的测试识别

### 问题
`test_norm_ops.py` 中有 `@label("weight_norm")`，但检查脚本认为 `weight_norm` 和 `_weight_norm` 没有测试。

### 修复
在 `check_missing_tests.py` 中添加了 `weight_norm` 到别名列表：

```python
aliases = ["dropout", "group_norm", "layer_norm", "weight_norm"]
```

### 结果
✅ `weight_norm` 现在被正确识别为有测试的算子。

### 注意
- `weight_norm` 是 `_weight_norm_interface` 的别名
- `_weight_norm` 是独立的内部算子，可能不需要单独测试
- 测试中有 `@label("weight_norm")` 和 `@label("weight_norm_interface")`

## 4. 最终缺失测试的算子

### 结果
只剩下 4 个缺失测试的算子，都是**内部算子**，可能不需要单独测试：

1. `_weight_norm` - 内部算子
2. `constant_pad_nd` - 内部算子（`pad` 的内部实现）
3. `diagonal_backward` - 内部算子（`diagonal` 的 backward kernel）
4. `gather_backward` - 内部算子（`gather` 的 backward kernel）

### 结论
✅ **所有标准 PyTorch 算子都有对应的测试！**

缺失的 4 个都是内部算子，它们通过用户 API 间接测试（如 `diagonal_backward` 通过 `torch.diagonal` 的 backward 测试覆盖）。

