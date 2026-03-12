# IMPL_INFO API 检查报告

## 检查目的
确保 `IMPL_INFO` 中注册的所有 API 名称都正确对应 PyTorch 的 `torch.ops.aten` 操作符，避免注册失败导致 fallback 到 PyTorch 的 kernel。

## 检查结果

### ✅ 总体情况
- **总 API 数量**: 290 个
- **有问题的 API**: 1 个
- **已修复**: 1 个

### ❌ 发现并修复的问题

#### 1. bitwise_and_.Tensor_ (已修复)
- **操作符**: `bitwise_and_`
- **原 API**: `bitwise_and_.Tensor_`
- **问题**: overload 名称 `Tensor_` 不存在
- **修复**: 改为 `bitwise_and_.Tensor`
- **状态**: ✅ 已修复

### ✅ 验证正确的 API

以下 API 已验证正确：

- `where.self_out` - ✅ overload 存在
- `log_softmax.int` - ✅ overload 存在
- `softmax.int` - ✅ overload 存在
- `_unique2` - ✅ base operator 存在
- `_to_copy` - ✅ base operator 存在
- `cross_entropy_loss` - ✅ base operator 存在

### 📝 自定义操作符（正常情况）

以下操作符是自定义的，不在 PyTorch 标准 `torch.ops.aten` 中，但如果你的实现正确注册了它们，这是正常的：

- `fused_add_rms_norm`
- `skip_layer_norm`
- `gelu_and_mul`
- `silu_and_mul`
- `apply_rotary_pos_emb`
- `concat_and_cache_mla`
- `flash_attention_forward`
- `flash_attn_varlen_func`
- `reshape_and_cache`
- `reshape_and_cache_flash`
- `rwkv_ka_fusion`
- `rwkv_mm_sparsity`
- `topk_softmax`
- `get_scheduler_metadata`
- `conv_depthwise2d`

这些操作符需要确保在你的 FlagGems 实现中正确注册，否则会 fallback 到 PyTorch。

## 建议

1. **定期检查**: 当添加新的操作符时，应该验证 API 名称是否正确
2. **测试注册**: 实际运行注册代码，检查是否有错误
3. **验证 fallback**: 运行 benchmark 时，确认是否真的使用了 Triton kernel
4. **自定义操作符**: 对于自定义操作符，确保在 FlagGems 中正确实现和注册

## 检查脚本

可以使用以下脚本定期检查：

```python
import torch
from flagbench.dataset.kernel_list import IMPL_INFO

for operator_name, api_list in IMPL_INFO.items():
    for api_name, _ in api_list:
        parts = api_name.split('.')
        base_op = getattr(torch.ops.aten, parts[0], None)
        if base_op and len(parts) > 1:
            if not hasattr(base_op, parts[1]):
                print(f"问题: {operator_name} -> {api_name}")
```

