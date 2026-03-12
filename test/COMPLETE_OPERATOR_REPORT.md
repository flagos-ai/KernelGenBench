# 83 个新增算子完整分类报告

生成时间: 2025-11-17

## 📊 总体统计

| 类别 | 数量 | 占比 | 状态 |
|------|------|------|------|
| **类别 1: 可测试** | 59 | 71.1% | ✅ 完美 |
| **类别 2: 自定义算子** | 14 | 16.9% | ✅ 正常 |
| **类别 3: 未注册** | 10 | 12.0% | ⚠️ 需要处理 |
| **总计** | **83** | **100%** | - |

---

## ✅ 类别 1: 可测试的算子（59 个）

**说明**: 这些算子同时在 IMPL_INFO 和 PYTORCH_OPERATORS 中注册。

**特点**:
- ✅ test_accuracy_ut.py 可以测试
- ✅ 你的新增测试函数也可以测试
- ✅ 注册完全正确

**列表**:
```
1. addcdiv                      -> torch.addcdiv
2. addcmul                      -> torch.addcmul
3. addr                         -> torch.addr
4. allclose                     -> torch.allclose
5. angle                        -> torch.angle
6. atan                         -> torch.atan
7. atan_                        -> torch.Tensor.atan_
8. batch_norm                   -> torch.batch_norm
9. bitwise_left_shift           -> torch.bitwise_left_shift
10. bitwise_right_shift         -> torch.bitwise_right_shift
11. celu                        -> torch.nn.functional.celu
12. clamp_min                   -> torch.clamp_min
13. clamp_min_                  -> torch.Tensor.clamp_min_
14. contiguous                  -> torch.Tensor.contiguous
15. conv3d                      -> torch.nn.functional.conv3d
16. cummax                      -> torch.cummax
17. cummin                      -> torch.cummin
18. dot                         -> torch.dot
19. elu_                        -> torch.nn.functional.elu_
20. embedding                   -> torch.embedding
21. exp2                        -> torch.exp2
22. exp2_                       -> torch.Tensor.exp2_
23. eye                         -> torch.eye
24. fill_                       -> torch.fill_
25. floor_divide                -> torch.floor_divide
26. floor_divide_               -> torch.Tensor.floor_divide_
27. gather                      -> torch.gather
28. gelu                        -> torch.nn.functional.gelu
29. glu                         -> torch.nn.functional.glu
30. index_add_                  -> torch.Tensor.index_add_
31. index_put_                  -> torch.index_put_
32. lerp                        -> torch.lerp
33. lerp_                       -> torch.Tensor.lerp_
34. linspace                    -> torch.linspace
35. log                         -> torch.log
36. log_softmax                 -> torch.log_softmax
37. logspace                    -> torch.logspace
38. masked_fill                 -> torch.masked_fill
39. masked_fill_                -> torch.Tensor.masked_fill_
40. max                         -> torch.max
41. max_pool2d                  -> torch.nn.functional.max_pool2d
42. min                         -> torch.min
43. nan_to_num                  -> torch.nan_to_num
44. polar                       -> torch.polar
45. rms_norm                    -> torch.rms_norm
46. scaled_dot_product_attention -> torch.nn.functional.scaled_dot_product_attention
47. scatter_                    -> torch.Tensor.scatter_
48. sigmoid                     -> torch.sigmoid
49. silu                        -> torch.nn.functional.silu
50. slice_scatter               -> torch.slice_scatter
51. softmax                     -> torch.softmax
52. softplus                    -> torch.nn.functional.softplus
53. sqrt                        -> torch.sqrt
54. sqrt_                       -> torch.Tensor.sqrt_
55. std                         -> torch.std
56. tanh                        -> torch.tanh
57. threshold                   -> torch.threshold
58. to                          -> torch.Tensor.to
59. trace                       -> torch.trace
```

---

## 🔧 类别 2: 自定义算子（14 个）

**说明**: 这些算子只在 IMPL_INFO 中，不在 PYTORCH_OPERATORS 中。

**特点**:
- ✅ 注册方式正确（它们不是 PyTorch 标准 API）
- ❌ test_accuracy_ut.py 无法测试（设计如此，不是 bug）
- ✅ 只能通过你的新增测试函数测试
- ✅ 无需额外处理

**列表**:
```
1. celu_                        (CELU inplace 版本)
2. concat_and_cache_mla         (MLA 缓存操作)
3. elu_backward                 (ELU 反向传播)
4. flash_attention_forward      (Flash Attention 前向)
5. flash_attn_varlen_func       (可变长度 Flash Attention)
6. fused_add_rms_norm           (融合 Add+RMS Norm)
7. get_scheduler_metadata       (调度器元数据)
8. index                        (索引操作)
9. max_pool2d_backward          (MaxPool2D 反向)
10. reshape_and_cache           (Reshape 缓存)
11. reshape_and_cache_flash     (Flash Reshape 缓存)
12. rwkv_ka_fusion              (RWKV KA 融合)
13. rwkv_mm_sparsity            (RWKV MM 稀疏)
14. topk_softmax                (TopK Softmax)
```

**对应测试文件**:
- test_attention_ops.py: flash_attention_forward, flash_attn_varlen_func, concat_and_cache_mla, reshape_and_cache, reshape_and_cache_flash
- test_special_ops.py: get_scheduler_metadata
- test_norm_ops.py: fused_add_rms_norm
- test_unary_pointwise_ops.py: celu_
- test_blas_ops.py: max_pool2d_backward
- 其他: rwkv_ka_fusion, rwkv_mm_sparsity, topk_softmax, index, elu_backward

---

## ❌ 类别 3: 未注册的算子（10 个）

**说明**: 这些算子在 IMPL_INFO 和 PYTORCH_OPERATORS 中都没有找到。

**特点**:
- ❌ 两边都没有注册
- ⚠️ 需要补充到 kernel_list.py
- ⚠️ 或者检查是否使用了不同的命名

**详细列表**:

| 序号 | 算子名 | 可能原因 | 建议操作 |
|------|--------|----------|----------|
| 1 | apply_rotary_pos_emb | 命名不一致 | 检查实际函数名 |
| 2 | conv2d | 可能已注册为其他名称 | 检查是否为 torch.nn.functional.conv2d |
| 3 | conv_depthwise2d | 自定义算子 | 需要注册 |
| 4 | dropout | 可能已注册 | 检查 torch.nn.functional.dropout |
| 5 | gelu_and_mul | 融合算子 | 需要注册 |
| 6 | group_norm | 可能已注册 | 检查 torch.nn.functional.group_norm |
| 7 | layer_norm | 可能已注册 | 检查 torch.nn.functional.layer_norm |
| 8 | silu_and_mul | 融合算子 | 需要注册 |
| 9 | skip_layer_norm | 自定义算子 | 需要注册 |
| 10 | weight_norm | 可能已注册 | 检查 torch.nn.utils.weight_norm |

---

## 🎯 目前不能被 test_accuracy_ut.py 测试的算子

**总计**: 24 个

### 1. 自定义算子（14 个）- 正常情况
这些算子需要通过你的新增测试函数测试（已完成 ✅）
```
celu_, concat_and_cache_mla, elu_backward, 
flash_attention_forward, flash_attn_varlen_func, 
fused_add_rms_norm, get_scheduler_metadata, index, 
max_pool2d_backward, reshape_and_cache, 
reshape_and_cache_flash, rwkv_ka_fusion, 
rwkv_mm_sparsity, topk_softmax
```

### 2. 未注册算子（10 个）- 需要处理 ⚠️
这些算子需要补充注册或确认命名
```
apply_rotary_pos_emb, conv2d, conv_depthwise2d, 
dropout, gelu_and_mul, group_norm, layer_norm, 
silu_and_mul, skip_layer_norm, weight_norm
```

---

## 📋 后续行动计划

### 优先级 1: 确认已存在但命名不同的算子
可能这些算子已经注册，但使用了不同的键名：
```bash
# 检查这些算子在 kernel_list.py 中的实际注册名
grep -i "conv2d" src/flagbench/dataset/kernel_list.py
grep -i "dropout" src/flagbench/dataset/kernel_list.py
grep -i "group_norm" src/flagbench/dataset/kernel_list.py
grep -i "layer_norm" src/flagbench/dataset/kernel_list.py
grep -i "weight_norm" src/flagbench/dataset/kernel_list.py
grep -i "rotary" src/flagbench/dataset/kernel_list.py
```

### 优先级 2: 补充真正缺失的自定义算子
如果确认这些是新实现的算子，需要在 kernel_list.py 中添加：
```python
IMPL_INFO = {
    # ... 现有条目
    "apply_rotary_pos_emb": [("apply_rotary_pos_emb", Autograd.disable)],
    "conv_depthwise2d": [("conv_depthwise2d", Autograd.disable)],
    "gelu_and_mul": [("gelu_and_mul", Autograd.disable)],
    "silu_and_mul": [("silu_and_mul", Autograd.disable)],
    "skip_layer_norm": [("skip_layer_norm", Autograd.disable)],
}
```

### 优先级 3: 更新测试工具
修改 test_accuracy_all.py 支持测试自定义算子（可选）

---

## 🎉 总结

### 当前状态
- ✅ **88.0%** (73/83) 的算子已正确处理
- ⚠️ **12.0%** (10/83) 的算子需要确认或补充注册

### 测试覆盖
- **test_accuracy_ut.py 可测试**: 59 个 (71.1%)
- **专门测试文件测试**: 14 个 (16.9%)
- **暂时无法测试**: 10 个 (12.0%)

### 你的工作价值
✅ 你新增的 101 个测试函数覆盖了：
- 59 个可测试的标准算子（增强测试）
- 14 个自定义算子（填补空白）
- 10 个待注册算子（为将来做准备）

**总体评价**: 注册架构正确，测试覆盖完善，只需补充 10 个算子的注册即可达到 100% 覆盖！
