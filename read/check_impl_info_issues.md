# IMPL_INFO API 检查结果

## 问题总结

### 1. 确认有问题的 API（需要修复）

#### bitwise_and_.Tensor_
- **操作符**: `bitwise_and_`
- **当前 API**: `bitwise_and_.Tensor_`
- **问题**: overload 名称 `Tensor_` 可能不正确
- **建议**: 检查应该是 `bitwise_and_.Tensor` 还是其他格式

#### where.self_out
- **操作符**: `where`
- **当前 API**: `where.self_out`
- **问题**: overload 名称格式可能不正确
- **建议**: 检查应该是 `where.self` 还是 `where.out`

### 2. 自定义操作符（不在 PyTorch 标准 API 中，这是正常的）

这些操作符是自定义的，不在 `torch.ops.aten` 中，但如果你的实现正确注册了它们，这是正常的：

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

### 3. 需要验证的 API（可能正确，但需要确认）

这些 API 在 PyTorch 中存在，但格式可能需要确认：

- `log_softmax.int` - 需要确认 overload 名称
- `softmax.int` - 需要确认 overload 名称
- `_unique2` - 需要确认这是正确的 API 名称
- `_to_copy` - 需要确认这是正确的 API 名称
- `cross_entropy_loss` - 需要确认这是正确的 API 名称（vs `cross_entropy`）

## 建议的检查步骤

1. **验证 overload 名称**: 对于有 `.` 的 API，需要确认 overload 名称是否正确
2. **测试注册**: 实际运行注册代码，看是否有错误
3. **检查 fallback**: 运行 benchmark，确认是否真的使用了 Triton kernel 而不是 PyTorch kernel

