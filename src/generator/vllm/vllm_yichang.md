# vLLM 算子异常记录

## 1. swap_blocks

**问题**：加速比异常，num_blocks=512 时 16x，num_blocks=32 时 0.5x

**原因**：
- testfunc 中 `dst_perf.clone()` 在 lambda 里面，每次 benchmark 都会执行 clone
- Triton 实现时间几乎恒定（~0.24ms），没有随数据量缩放

**建议修复**：
```python
# 把 clone 移到 lambda 外面
dst_baseline = dst_perf.clone()
dst_triton = dst_perf.clone()

ms_baseline = triton.testing.do_bench(
    lambda: flagbench.baseline.swap_blocks(src_perf, dst_baseline, block_mapping),
    warmup=25, rep=100
)
ms_triton = triton.testing.do_bench(
    lambda: flagbench.triton.swap_blocks(src_perf, dst_triton, block_mapping),
    warmup=25, rep=100
)
```

**文件位置**：`src/flagbench/accuracy/vllm13/test_swap_blocks.py`

test_apply_repetition_penalties

test_gptq_marlin_repack

test_awq_marlin_repack

test_gptq_marlin_moe_repack

test_awq_marlin_moe_repack

test_allspark_repack_weight

test_swap_blocks

test_marlin_int4_fp8_preprocess