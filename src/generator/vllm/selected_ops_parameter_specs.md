# vLLM 核心算子参数规格文档

本文档包含9个核心vLLM算子的完整参数类型信息，用于测试函数的参数范围验证。

## 算子列表

1. `selective_scan_fwd` - 选择性扫描前向传播
2. `paged_attention_v1` - 页面注意力机制v1
3. `scaled_fp8_quant` - 缩放FP8量化
4. `rotary_embedding` - 旋转位置嵌入
5. `fused_add_rms_norm` - 融合的加法和RMS归一化
6. `silu_and_mul_scaled_fp4_experts_quant` - SiLU激活和乘法缩放FP4专家量化
7. `fused_qk_norm_rope` - 融合的Q/K归一化和RoPE
8. `scaled_int8_quant` - 缩放INT8量化
9. `cutlass_scaled_mm` - Cutlass缩放矩阵乘法

---

## 1. selective_scan_fwd

**签名:**
```python
selective_scan_fwd(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D_: torch.Tensor | None, z_: torch.Tensor | None, delta_bias_: torch.Tensor | None, delta_softplus: bool, query_start_loc: torch.Tensor | None, cache_indices: torch.Tensor | None, has_initial_state: torch.Tensor | None, ssm_states: torch.Tensor, pad_slot_id: int, block_size: int = 1024, block_idx_first_scheduled_token: torch.Tensor | None = None, block_idx_last_scheduled_token: torch.Tensor | None = None, initial_state_idx: torch.Tensor | None = None)
```

**输入参数:**
- `u: torch.Tensor` - 输入张量u
- `delta: torch.Tensor` - delta张量
- `A: torch.Tensor` - A矩阵
- `B: torch.Tensor` - B矩阵
- `C: torch.Tensor` - C矩阵
- `D_: torch.Tensor | None` - 可选的D张量
- `z_: torch.Tensor | None` - 可选的z张量
- `delta_bias_: torch.Tensor | None` - 可选的delta偏置
- `delta_softplus: bool` - 是否使用softplus激活
- `query_start_loc: torch.Tensor | None` - 可选的查询起始位置
- `cache_indices: torch.Tensor | None` - 可选的缓存索引
- `has_initial_state: torch.Tensor | None` - 可选的初始状态标志
- `ssm_states: torch.Tensor` - SSM状态
- `pad_slot_id: int` - 填充槽ID
- `block_size: int = 1024` - 块大小，默认1024
- `block_idx_first_scheduled_token: torch.Tensor | None = None` - 可选的第一个调度token的块索引
- `block_idx_last_scheduled_token: torch.Tensor | None = None` - 可选的最后一个调度token的块索引
- `initial_state_idx: torch.Tensor | None = None` - 可选的初始状态索引

**输出参数:** 无返回值（原地操作）

**vLLM API调用:**
```python
from vllm import _custom_ops
_custom_ops.selective_scan_fwd(...)
```

---

## 2. paged_attention_v1

**签名:**
```python
paged_attention_v1(out: torch.Tensor, query: torch.Tensor, key_cache: torch.Tensor, value_cache: torch.Tensor, num_kv_heads: int, scale: float, block_tables: torch.Tensor, seq_lens: torch.Tensor, block_size: int, max_seq_len: int, alibi_slopes: torch.Tensor | None, kv_cache_dtype: str, k_scale: torch.Tensor, v_scale: torch.Tensor, tp_rank: int = 0, blocksparse_local_blocks: int = 0, blocksparse_vert_stride: int = 0, blocksparse_block_size: int = 64, blocksparse_head_sliding_step: int = 0) -> None
```

**输入参数:**
- `out: torch.Tensor` - 输出张量
- `query: torch.Tensor` - 查询张量
- `key_cache: torch.Tensor` - Key缓存
- `value_cache: torch.Tensor` - Value缓存
- `num_kv_heads: int` - KV头数量
- `scale: float` - 缩放因子
- `block_tables: torch.Tensor` - 块表
- `seq_lens: torch.Tensor` - 序列长度
- `block_size: int` - 块大小
- `max_seq_len: int` - 最大序列长度
- `alibi_slopes: torch.Tensor | None` - 可选的ALiBi斜率
- `kv_cache_dtype: str` - KV缓存数据类型
- `k_scale: torch.Tensor` - K缩放因子
- `v_scale: torch.Tensor` - V缩放因子
- `tp_rank: int = 0` - 张量并行rank，默认0
- `blocksparse_local_blocks: int = 0` - 块稀疏本地块数，默认0
- `blocksparse_vert_stride: int = 0` - 块稀疏垂直步长，默认0
- `blocksparse_block_size: int = 64` - 块稀疏块大小，默认64
- `blocksparse_head_sliding_step: int = 0` - 块稀疏头滑动步长，默认0

**输出参数:** `None`（原地操作到out张量）

**vLLM API调用:**
```python
from vllm import _custom_ops
_custom_ops.paged_attention_v1(...)
```

---

## 3. scaled_fp8_quant

**签名:**
```python
scaled_fp8_quant(input: torch.Tensor, scale: torch.Tensor | None = None, num_token_padding: int | None = None, scale_ub: torch.Tensor | None = None, use_per_token_if_dynamic: bool = False, output: torch.Tensor | None = None, group_shape: tuple[int, int] | None = None) -> tuple
```

**输入参数:**
- `input: torch.Tensor` - 输入张量（必须是2D: [M, N]）
- `scale: torch.Tensor | None = None` - 可选的缩放因子
  - 0D或[1]: 每张量缩放
  - 1D: 需要明确的group_shape来区分per-channel vs per-token
  - 2D [M/group_m, N/group_n]: 组缩放
- `num_token_padding: int | None = None` - 可选的token填充数量
- `scale_ub: torch.Tensor | None = None` - 可选的动态per-token情况下的缩放因子上限
- `use_per_token_if_dynamic: bool = False` - 动态量化时是否使用per-token
- `output: torch.Tensor | None = None` - 可选的输出张量
- `group_shape: tuple[int, int] | None = None` - 可选的组形状元组(group_m, group_n)

**输出参数:** `tuple` - (量化后的FP8张量, 缩放因子)

**文档:**
量化输入张量到FP8并返回量化张量和缩放因子。

支持静态和动态量化：如果提供scale则使用静态缩放，否则动态确定scale。也允许可选的输出张量填充以便下游内核受益。

**vLLM API调用:**
```python
from vllm import _custom_ops
_custom_ops.scaled_fp8_quant(...)
```

---

## 4. rotary_embedding

**签名:**
```python
rotary_embedding(positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor | None, head_size: int, cos_sin_cache: torch.Tensor, is_neox: bool) -> None
```

**输入参数:**
- `positions: torch.Tensor` - 位置张量
- `query: torch.Tensor` - 查询张量
- `key: torch.Tensor | None` - 可选的Key张量
- `head_size: int` - 头大小
- `cos_sin_cache: torch.Tensor` - 余弦正弦缓存
- `is_neox: bool` - 是否为NeoX风格

**输出参数:** `None`（原地操作）

**vLLM API调用:**
```python
from vllm import _custom_ops
_custom_ops.rotary_embedding(...)
```

---

## 5. fused_add_rms_norm

**签名:**
```python
fused_add_rms_norm(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float) -> None
```

**输入参数:**
- `input: torch.Tensor` - 输入张量
- `residual: torch.Tensor` - 残差张量
- `weight: torch.Tensor` - 权重张量
- `epsilon: float` - epsilon值

**输出参数:** `None`（原地操作到input张量）

**vLLM API调用:**
```python
from vllm import _custom_ops
_custom_ops.fused_add_rms_norm(...)
```

---

## 6. silu_and_mul_scaled_fp4_experts_quant

**签名:**
```python
silu_and_mul_scaled_fp4_experts_quant(input_tensor: torch.Tensor, input_global_scale: torch.Tensor, expert_offsets: torch.Tensor, blockscale_offsets: torch.Tensor, topk: int) -> tuple
```

**输入参数:**
- `input_tensor: torch.Tensor` - 输入张量，gate || up布局 [m_topk, k*2]
- `input_global_scale: torch.Tensor` - 每专家缩放因子 [n_experts]
- `expert_offsets: torch.Tensor` - 专家偏移张量 [n_experts+1]
- `blockscale_offsets: torch.Tensor` - 块缩放偏移张量 [n_experts+1]
- `topk: int` - 选择的top-k专家数量

**输出参数:** `tuple` - (NVFP4量化张量 [m_topk, k/2], FP8-E4M3块缩放张量)

**文档:**
MoE中间激活的融合SiLU+Mul+NVFP4量化。

**vLLM API调用:**
```python
from vllm import _custom_ops
_custom_ops.silu_and_mul_scaled_fp4_experts_quant(...)
```

---

## 7. fused_qk_norm_rope

**签名:**
```python
fused_qk_norm_rope(qkv: torch.Tensor, num_heads_q: int, num_heads_k: int, num_heads_v: int, head_dim: int, eps: float, q_weight: torch.Tensor, k_weight: torch.Tensor, cos_sin_cache: torch.Tensor, is_neox: bool, position_ids: torch.Tensor) -> None
```

**输入参数:**
- `qkv: torch.Tensor` - QKV张量
- `num_heads_q: int` - Q头数量
- `num_heads_k: int` - K头数量
- `num_heads_v: int` - V头数量
- `head_dim: int` - 头维度
- `eps: float` - epsilon值
- `q_weight: torch.Tensor` - Q权重
- `k_weight: torch.Tensor` - K权重
- `cos_sin_cache: torch.Tensor` - 余弦正弦缓存
- `is_neox: bool` - 是否为NeoX风格
- `position_ids: torch.Tensor` - 位置ID

**输出参数:** `None`（原地操作）

**vLLM API调用:**
```python
from vllm import _custom_ops
_custom_ops.fused_qk_norm_rope(...)
```

---

## 8. scaled_int8_quant

**签名:**
```python
scaled_int8_quant(input: torch.Tensor, scale: torch.Tensor | None = None, azp: torch.Tensor | None = None, symmetric: bool = True) -> tuple
```

**输入参数:**
- `input: torch.Tensor` - 输入张量
- `scale: torch.Tensor | None = None` - 可选的缩放因子，不提供时使用动态per-token量化
- `azp: torch.Tensor | None = None` - 可选的零点，用于非对称量化
- `symmetric: bool = True` - 是否使用对称量化

**输出参数:** `tuple` - (INT8张量, 缩放因子, 可选的零点)

**文档:**
将输入张量量化到int8并返回量化张量、缩放因子和可选的零点。

**vLLM API调用:**
```python
from vllm import _custom_ops
_custom_ops.scaled_int8_quant(...)
```

---

## 9. cutlass_scaled_mm

**签名:**
```python
cutlass_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, out_dtype: torch.dtype, bias: torch.Tensor | None = None) -> torch.Tensor
```

**输入参数:**
- `a: torch.Tensor` - A矩阵
- `b: torch.Tensor` - B矩阵
- `scale_a: torch.Tensor` - A的缩放因子
- `scale_b: torch.Tensor` - B的缩放因子
- `out_dtype: torch.dtype` - 输出数据类型
- `bias: torch.Tensor | None = None` - 可选的偏置

**输出参数:** `torch.Tensor` - 结果张量

**文档:**
`cutlass_scaled_mm`实现了融合版本的`output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`，其中scale_a * a和scale_b * b使用numpy风格的广播实现。

为了支持DeepSeek V3中发现的块级缩放，我们还支持扩展的"group"广播规则。

**vLLM API调用:**
```python
from vllm import _custom_ops
_custom_ops.cutlass_scaled_mm(...)
```

---

## 注意事项

1. 所有算子都需要先导入vLLM: `from vllm import _custom_ops`
2. 部分算子进行原地操作（inplace），输出参数为None
3. 量化相关算子返回tuple，包含量化结果和缩放因子
4. 参数类型标注使用Python 3.10+的联合类型语法（`|`）
5. 可选参数使用`None`作为默认值
6. 张量参数需要确保正确的形状和数据类型以避免vLLM不支持的范围错误