# 别名映射的原因说明

## 问题：是不是几个 key 对应一个 test？

**答案：不是。** 实际情况是：**一个底层 API 对应两个测试 label（上层名称和底层名称）**

## 具体情况

### 例子：dropout

```python
# 测试文件中
@label("dropout")          # 上层名称（用户友好）
@label("native_dropout")   # 底层名称（实际实现）
def test_accuracy_dropout(...):
    ref_out = torch.nn.functional.dropout(...)  # 用户调用的 API
    # 实际底层调用: torch.ops.aten.native_dropout
```

### 映射关系

```
测试 label: "dropout" 或 "native_dropout"
    ↓
IMPL_INFO["dropout"] = IMPL_INFO["native_dropout"]
    ↓
实际注册: torch.ops.aten.native_dropout
```

## 原因：PyTorch 的两层 API 结构

### 1. 上层 API（用户调用）
- `torch.nn.functional.dropout()`
- `torch.group_norm()`
- `torch.layer_norm()`
- `torch._weight_norm()`

### 2. 底层 API（实际实现）
- `torch.ops.aten.native_dropout`
- `torch.ops.aten.native_group_norm`
- `torch.ops.aten.native_layer_norm`
- `torch.ops.aten._weight_norm_interface`

## 为什么需要别名？

1. **测试中使用用户友好的名称**：
   - `@label("dropout")` 而不是 `@label("native_dropout")`
   - 这样测试更容易理解

2. **但注册时必须用底层 API 名称**：
   - `lib.impl("native_dropout", fn, device_key)`
   - 因为 PyTorch 的注册机制是基于底层 API

3. **别名映射的作用**：
   - `IMPL_INFO["dropout"] = IMPL_INFO["native_dropout"]`
   - 让测试中的 "dropout" label 能找到 "native_dropout" 的注册信息

## 完整的映射关系

| 测试 label（上层） | 别名映射 | 实际注册（底层） | PyTorch API |
|-------------------|---------|-----------------|-------------|
| `dropout` | → | `native_dropout` | `torch.nn.functional.dropout` → `torch.ops.aten.native_dropout` |
| `group_norm` | → | `native_group_norm` | `torch.group_norm` → `torch.ops.aten.native_group_norm` |
| `layer_norm` | → | `native_layer_norm` | `torch.layer_norm` → `torch.ops.aten.native_layer_norm` |
| `weight_norm` | → | `_weight_norm_interface` | `torch._weight_norm` → `torch.ops.aten._weight_norm_interface` |

## 总结

- **不是**"几个 key 对应一个 test"
- **而是**"一个底层 API 对应两个测试 label（上层名称和底层名称）"
- **原因**：PyTorch 有上层 API 和底层 API 两层结构
- **别名的作用**：让测试中的上层名称能找到底层 API 的注册信息

