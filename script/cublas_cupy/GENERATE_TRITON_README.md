# cuBLAS Triton Kernel 生成指南

## 概述

`generate_triton_cublas.py` 脚本用于批量生成 cuBLAS 操作的 Triton kernel 实现。它会：

1. 从 `cublas_ops.json` 读取 cuBLAS 函数签名
2. 调用 LLM 为每个函数生成 Triton kernel 实现
3. 生成的 Triton kernel 应该匹配 cuBLAS 的性能和功能
4. 自动保存到指定输出目录

## 使用方法

### 环境准备

```bash
# 激活环境（必须！）
source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench

# 进入项目目录
cd /share/project/zpy/flagbench
```

### 基本用法

#### 1. 为单个操作生成 Triton kernel

```bash
# 为 GEMM 操作生成 Triton kernel
python script/cublas_cupy/generate_triton_cublas.py --name gemm

# 输出目录：output_triton_cublas/triton_cublas_{model}_{timestamp}/
```

这会生成 5 个 GEMM 变体的 Triton kernel：
- `gemm.py` (通用 GEMM kernel，可能包含多个 dtype 版本)

#### 2. 为所有操作生成 Triton kernel

```bash
# 生成全部 219 个有效 BLAS 函数的 Triton kernel
python script/cublas_cupy/generate_triton_cublas.py --name all

# ⚠️ 注意：这会调用 LLM 219 次，可能需要较长时间和费用
```

### 高级选项

#### 使用不同的 LLM 模型

```bash
# 使用 DeepSeek 模型
python script/cublas_cupy/generate_triton_cublas.py \
    --name gemm \
    --server-type deepseek \
    --model-name deepseek-coder

# 使用 OpenAI
python script/cublas_cupy/generate_triton_cublas.py \
    --name gemm \
    --server-type openai \
    --model-name gpt-4
```

#### 自定义输出目录

```bash
python script/cublas_cupy/generate_triton_cublas.py \
    --name gemm \
    --output-dir /path/to/my/output
```

#### 调整生成参数

```bash
python script/cublas_cupy/generate_triton_cublas.py \
    --name gemm \
    --temperature 0.0 \        # 温度（0.0 = 确定性）
    --max-tokens 16384 \       # 最大 token 数
    --num-samples 3 \          # 为每个函数生成 3 个版本
    --num-workers 20 \         # 并行度
    --max-retries 5 \          # 失败重试次数
    --retry-delay 2.0          # 重试延迟（秒）
```

#### 批量生成常用操作

```bash
# Level 3 BLAS (矩阵-矩阵运算)
for op in gemm symm syrk syr2k trmm trsm; do
    python script/cublas_cupy/generate_triton_cublas.py --name $op
done

# Level 2 BLAS (矩阵-向量运算)
for op in gemv ger syr syr2 trmv trsv; do
    python script/cublas_cupy/generate_triton_cublas.py --name $op
done

# Level 1 BLAS (向量运算)
for op in axpy dot scal copy swap asum nrm2; do
    python script/cublas_cupy/generate_triton_cublas.py --name $op
done
```

## 输出结构

```
output_triton_cublas/
└── triton_cublas_deepseek-v3-0324_num_samples_1_temp_0.0_max_tokens_16384_20260119-123456/
    ├── triton_0/                    # 第一个样本
    │   ├── gemm.py                  # GEMM Triton kernel
    │   ├── axpy.py                  # AXPY Triton kernel
    │   └── ...
    ├── triton_1/                    # 第二个样本（如果 num_samples > 1）
    │   └── ...
    └── generation_summary.json      # 生成统计报告
```

## 生成的 Triton Kernel 格式

每个生成的 Triton kernel 应该包含：

```python
import triton
import triton.language as tl
import torch

@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Triton kernel for GEMM operation"""
    # Kernel implementation
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # ... compute matrix multiplication ...

def gemm(a, b, alpha=1.0, beta=0.0, c=None):
    """
    Triton wrapper for GEMM
    
    Implements: C = alpha * A @ B + beta * C
    """
    M, K = a.shape
    K, N = b.shape
    
    if c is None:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N'])
    )
    
    gemm_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
    )
    
    return c
```

## 与测试函数生成的区别

| 特性 | `generate_ut_cublas.py` | `generate_triton_cublas.py` |
|------|------------------------|----------------------------|
| **目的** | 生成测试函数 | 生成 Triton kernel 实现 |
| **输出** | 测试代码（含 baseline 和 Triton 调用） | Triton kernel + wrapper 函数 |
| **Baseline** | CuPy cuBLAS（作为 reference） | cuBLAS（作为性能目标） |
| **Generator** | `TestFuncGenerator` | `TritonKernelGenerator` |
| **Args 类型** | `TestFuncGenerateArgs` | `TritonKernelGenerateArgs` |
| **输出目录** | `output_ut_cublas/` | `output_triton_cublas/` |

## 技术细节

### 工作流程

1. **加载 schema**: 从 `cublas_ops.json` 读取 239 个函数签名
2. **过滤函数**: 排除非 BLAS 函数（logger、handle 等）→ 219 个有效函数
3. **创建参数**: 为每个函数构造 `TritonKernelGenerateArgs`
   - `triton_kernel_name` = 操作名（如 "gemm"）
   - `use_cublas` = True（标记使用 cuBLAS prompt）
   - `cublas_schema` = 完整函数信息
   - `input_args` / `output_args` = 从 schema 推断
4. **调用 LLM**: 通过 `CuBLASTritonKernelGenerator` 调用 `generate_prompt_for_cublas()`
5. **保存结果**: 生成的 Triton kernel 保存到 `output_triton_cublas/`

### Prompt 特点

生成的 prompt 包含：
- cuBLAS 函数签名和参数说明
- CuPy cuBLAS 参考实现（用于理解语义）
- Triton kernel 示例（GEMM 等）
- 性能优化要求（block size, 内存合并等）
- PyTorch 参考实现（如果有）
- Wiki 参考实现（如果有）

### 代码修改

1. **`src/generator/triton_kernel_generator.py`**
   - 添加 `use_cublas` 检查在 `generate_prompt()`
   - 添加 `generate_prompt_for_cublas()` 方法（~200 行）

2. **`script/cublas_cupy/generate_triton_cublas.py`**
   - 新脚本，类似 `generate_ut_cublas.py`
   - 使用 `TritonKernelGenerator` 而非 `TestFuncGenerator`
   - 创建 `TritonKernelGenerateArgs` 而非 `TestFuncGenerateArgs`

## 检查生成结果

### 查看生成统计

```bash
# 查看生成成功率
cat output_triton_cublas/*/generation_summary.json | python -m json.tool | grep -E "total|successful|failed|success_rate"
```

### 快速验证生成的代码

```python
# 检查生成的文件数量
ls -1 output_triton_cublas/triton_*/triton_0/*.py | wc -l

# 检查某个文件的内容
cat output_triton_cublas/triton_*/triton_0/gemm.py
```

### 测试 Triton kernel

```python
import torch
import sys
sys.path.insert(0, 'output_triton_cublas/triton_*/triton_0/')

from gemm import gemm

# 测试
A = torch.randn(128, 256, device='cuda', dtype=torch.float32)
B = torch.randn(256, 512, device='cuda', dtype=torch.float32)

# Triton implementation
C_triton = gemm(A, B)

# PyTorch reference
C_torch = torch.mm(A, B)

# 比较
assert torch.allclose(C_triton, C_torch, rtol=1e-5, atol=1e-5)
print("✓ Triton kernel matches PyTorch!")
```

## 常见问题

### Q1: 生成失败怎么办？

检查 `generation_summary.json` 中的错误信息：

```bash
cat output_triton_cublas/*/generation_summary.json | python -c "
import sys, json
data = json.load(sys.stdin)
for r in data['results']:
    if not r['success']:
        print(f\"Failed: {r['cublas_function']}: {r.get('error', 'Unknown')}\")"
```

### Q2: 生成的 kernel 性能如何？

生成的 Triton kernel 性能取决于：
- LLM 模型质量（DeepSeek, GPT-4 等）
- Prompt 中的优化提示
- 是否使用了 Wiki 参考实现

建议：生成后进行性能测试，与 cuBLAS 对比。

### Q3: 如何调试生成的 kernel？

使用 `--verbose` 标志查看详细日志：

```bash
python script/cublas_cupy/generate_triton_cublas.py \
    --name gemm \
    --verbose 2>&1 | tee generation.log
```

### Q4: 支持哪些数据类型？

与测试函数生成相同：
- `float16` (H): 7 个函数
- `float32` (S): 51 个函数
- `float64` (D): 49 个函数
- `complex64` (C): 61 个函数
- `complex128` (Z): 52 个函数

## 完整工作流示例

```bash
# 1. 激活环境
source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench
cd /share/project/zpy/flagbench

# 2. 生成 GEMM Triton kernel
python script/cublas_cupy/generate_triton_cublas.py --name gemm

# 3. 查看生成结果
ls -lh output_triton_cublas/triton_*/triton_0/gemm.py

# 4. 生成对应的测试函数
python script/cublas_cupy/generate_ut_cublas.py --name gemm

# 5. 运行测试验证 Triton kernel 正确性
pytest output_ut_cublas/ut_*/ut_0/test_gemm_*.py -v

# 6. 性能对比（使用 FlagBench 的 benchmark 工具）
# TODO: 整合到 FlagBench 主流程
```

## 下一步工作

生成 Triton kernel 后，你可以：

1. **验证正确性**: 运行生成的测试函数，确保结果正确
2. **性能对比**: 使用 FlagBench 的 benchmark 工具对比 Triton vs cuBLAS 性能
3. **优化 kernel**: 根据性能结果调整 block size、内存访问模式等
4. **集成到 FlagBench**: 将生成的 Triton kernel 整合到 FlagGems 或其他框架

## 参考资料

- Triton 文档: https://triton-lang.org/
- cuBLAS 文档: https://docs.nvidia.com/cuda/cublas/
- CuPy cuBLAS 封装: https://docs.cupy.dev/en/stable/reference/cublas.html
- FlagBench 架构: `AGENTS.md`
