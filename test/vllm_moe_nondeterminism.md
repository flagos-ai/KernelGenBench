# vllm moe_align_block_size 非确定性问题

## 问题描述

`vllm._custom_ops.moe_align_block_size` 的 C++ 实现是**非确定性的**。
同一个函数、同一个输入，调用两次，`sorted_token_ids` 结果不一致。

## 测试证据

测试脚本：`test/test_vllm_moe_determinism.py`

运行命令：
```bash
source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench
cd /share/project/zpy/flagbench
python test/test_vllm_moe_determinism.py
```

### 测试结果

| config | sorted_token_ids | experts_ids | num_tokens_post_pad |
|--------|-----------------|-------------|-------------------|
| tokens=128, experts=4, block=16, topk=2 | 一致 | 一致 | 一致 |
| tokens=1024, experts=8, block=16, topk=2 | **不一致 (1926/2176)** | 一致 | 一致 |
| tokens=4096, experts=16, block=64, topk=4 | **不一致 (16196/17408)** | 一致 | 一致 |

## 原因分析

`moe_align_block_size` 的语义是"把 token 按 expert 分组并对齐到 block 边界"。
同一个 expert 内的 token 排列顺序不唯一，C++ 实现内部的并行执行导致每次调用排列不同。

- `sorted_token_ids`：非确定性（同 expert 内 token 顺序不固定）
- `experts_ids`：确定性
- `num_tokens_post_pad`：确定性
