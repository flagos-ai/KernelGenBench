# API 注册修正说明

## 问题发现

用户发现了一个重要问题：`addmm_out` 和 `addmv_out` 并不是真正的 PyTorch API，而只是测试函数的名字。

## 问题分析

### 错误的理解
最初认为 `addmm_out` 和 `addmv_out` 是独立的 API，需要单独注册。

### 正确的理解
1. **测试函数命名**：`test_accuracy_addmm_out` 和 `test_accuracy_addmv_out` 是测试函数的名字
2. **实际 API 调用**：这些测试实际调用的是：
   ```python
   torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta, out=out)
   torch.addmv(input, mat, vec, beta=beta, alpha=alpha, out=out)
   ```
3. **ATen 签名**：在 PyTorch 内部，这些对应的 ATen 操作符是：
   - `aten::addmm.out`
   - `aten::addmv.out`

### PyTorch 操作符重载机制
```python
>>> torch.ops.aten.addmm.overloads()
['default', 'out']

>>> torch.ops.aten.addmm.out._schema
aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, 
                Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
```

## 修正方案

### 修改前
```python
# 错误的注册方式
"addmm": [("addmm", Autograd.disable)],
"addmv": [("addmv", Autograd.disable)],
...
"addmm_out": [("addmm.out", Autograd.disable)],  # ❌ 错误：这不是独立的 API
"addmv_out": [("addmv.out", Autograd.disable)],  # ❌ 错误：这不是独立的 API
```

### 修改后
```python
# 正确的注册方式
"addmm": [("addmm", Autograd.disable), ("addmm.out", Autograd.disable)],
"addmv": [("addmv", Autograd.disable), ("addmv.out", Autograd.disable)],
# 不需要单独注册 addmm_out 和 addmv_out
```

## 其他 API 检查结果

### 确认存在于标准 PyTorch 的 API
以下 API 之前被误认为是"自定义"的，实际上都存在于标准 PyTorch 中：
- ✅ `angle` - `torch.angle` 存在
- ✅ `log` - `torch.log` 存在（虽然代码中可能有自定义实现）
- ✅ `contiguous` - `torch.Tensor.contiguous` 存在
- ✅ `celu_` - `torch.celu_` 存在（inplace 版本）

### 真正的自定义/融合 API（10 个）
只有以下 API 不在标准 PyTorch 中，是自定义实现：
1. `concat_and_cache_mla`
2. `flash_attention_forward`
3. `flash_attn_varlen_func`
4. `fused_add_rms_norm`
5. `get_scheduler_metadata`
6. `reshape_and_cache`
7. `reshape_and_cache_flash`
8. `rwkv_ka_fusion`
9. `rwkv_mm_sparsity`
10. `topk_softmax`

## 最终统计

### IMPL_INFO 字典
- **总条目数**: 220 个（从 222 减少到 220）
- **修正内容**:
  - 删除了 `addmm_out` 和 `addmv_out` 的独立注册（-2）
  - 在 `addmm` 和 `addmv` 中添加了 `.out` 变体

### PYTORCH_OPERATORS 字典
- **总条目数**: 201 个（未变化）
- **原因**: `addmm` 和 `addmv` 本来就已经在字典中

## 测试函数与 API 的映射

| 测试函数 | 使用的 API | IMPL_INFO 中的键 |
|---------|-----------|----------------|
| `test_accuracy_addmm_out` | `torch.addmm(..., out=out)` | `addmm` (包含 `addmm.out`) |
| `test_accuracy_addmv_out` | `torch.addmv(..., out=out)` | `addmv` (包含 `addmv.out`) |

## 经验总结

1. **测试函数名 ≠ API 名**: 测试函数可以随意命名，不要被名字误导
2. **查看实际代码**: 必须查看测试函数内部实际调用了什么 API
3. **理解 ATen 签名**: PyTorch 的 `out=` 参数对应 ATen 的 `.out` 重载
4. **验证 API 存在性**: 使用 `hasattr()` 验证 API 是否真的存在于 PyTorch 中

## 文件修改

**文件**: `/share/project/zpy/flagbench/src/flagbench/dataset/kernel_list.py`

**修改内容**:
1. 第 18-19 行：更新 `addmm` 和 `addmv` 的注册，添加 `.out` 变体
2. 第 322-324 行：删除 `addmm_out` 和 `addmv_out` 的错误注册，添加说明注释
3. 第 607-610 行：更新注释，移除错误的"自定义 API"标记

---
更新时间: 2025-11-17
状态: ✅ 已修正
