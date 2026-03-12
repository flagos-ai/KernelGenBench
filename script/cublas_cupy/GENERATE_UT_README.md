# cuBLAS 测试函数批量生成指南

## 概述

`generate_ut_cublas.py` 脚本用于批量生成 cuBLAS 操作的测试函数。它会：

1. 从 `cublas_ops.json` 读取 cuBLAS 函数签名
2. 调用 LLM 为每个函数生成测试代码
3. 测试代码包含：CuPy baseline（cuBLAS 参考实现） vs Triton 实现
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

#### 1. 为单个操作生成测试（推荐先测试）

```bash
# 为 GEMM 操作生成测试函数
python script/cublas_cupy/generate_ut_cublas.py --name gemm

# 输出目录：output_ut_cublas/ut_cublas_{model}_{timestamp}/
```

这会生成 5 个 GEMM 变体的测试函数：
- `test_gemm_float16.py` (cublasHgemm)
- `test_gemm_float32.py` (cublasSgemm_v2)
- `test_gemm_float64.py` (cublasDgemm_v2)
- `test_gemm_complex64.py` (cublasCgemm_v2)
- `test_gemm_complex128.py` (cublasZgemm_v2)

#### 2. 为所有操作生成测试

```bash
# 生成全部 219 个有效 BLAS 函数的测试
python script/cublas_cupy/generate_ut_cublas.py --name all

# ⚠️ 注意：这会调用 LLM 219 次，可能需要较长时间和费用
```

### 高级选项

#### 使用不同的 LLM 模型

```bash
# 使用 DeepSeek 模型
python script/cublas_cupy/generate_ut_cublas.py \
    --name gemm \
    --server-type deepseek \
    --model-name deepseek-coder

# 使用 OpenAI
python script/cublas_cupy/generate_ut_cublas.py \
    --name gemm \
    --server-type openai \
    --model-name gpt-4
```

#### 自定义输出目录

```bash
python script/cublas_cupy/generate_ut_cublas.py \
    --name gemm \
    --output-dir /path/to/my/output
```

#### 调整生成参数

```bash
python script/cublas_cupy/generate_ut_cublas.py \
    --name gemm \
    --temperature 0.0 \        # 温度（0.0 = 确定性）
    --max-tokens 16384 \       # 最大 token 数
    --num-samples 3 \          # 为每个函数生成 3 个版本
    --num-workers 20 \         # 并行度（越高越快，但可能触发 rate limit）
    --max-retries 5 \          # 失败重试次数
    --retry-delay 2.0          # 重试延迟（秒）
```

#### 批量生成常用操作

```bash
# Level 3 BLAS (矩阵-矩阵运算)
for op in gemm symm syrk syr2k trmm trsm; do
    python script/cublas_cupy/generate_ut_cublas.py --name $op
done

# Level 2 BLAS (矩阵-向量运算)
for op in gemv ger syr syr2 trmv trsv; do
    python script/cublas_cupy/generate_ut_cublas.py --name $op
done

# Level 1 BLAS (向量运算)
for op in axpy dot scal copy swap asum nrm2; do
    python script/cublas_cupy/generate_ut_cublas.py --name $op
done
```

## 输出结构

```
output_ut_cublas/
└── ut_cublas_deepseek-v3-0324_num_samples_1_temp_0.0_max_tokens_16384_20260119-123456/
    ├── ut_0/                          # 第一个样本
    │   ├── test_gemm_float32.py       # GEMM float32 测试
    │   ├── test_gemm_float64.py       # GEMM float64 测试
    │   └── ...
    ├── ut_1/                          # 第二个样本（如果 num_samples > 1）
    │   └── ...
    └── generation_summary.json        # 生成统计报告
```

## 生成的测试函数格式

每个生成的测试函数大致如下：

```python
import torch
import cupy as cp
from torch.utils.dlpack import to_dlpack, from_dlpack

@label("gemm")
@parametrize("M, N, K", [(128, 256, 64), (512, 512, 256)])
@parametrize("dtype", [torch.float32])
def test_gemm_float32(M, N, K, dtype):
    # 初始化输入张量
    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')
    
    # CuPy baseline（cuBLAS 参考实现）
    A_cp = cp.from_dlpack(to_dlpack(A))
    B_cp = cp.from_dlpack(to_dlpack(B))
    ref_out_cp = cp.dot(A_cp, B_cp)
    ref_out = from_dlpack(ref_out_cp.toDlpack())
    
    # Triton 实现
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.mm(A, B)
    
    # 比较结果
    assert_close(act_out, ref_out, dtype=dtype)
```

## 统计信息

### 当前支持的函数

- **总函数数**: 239（来自 cublas_v2.h）
- **有效 BLAS 函数**: 219（已过滤掉 logger、handle 等非 BLAS 函数）
- **Level 1 BLAS**: ~34 个（向量操作）
- **Level 2 BLAS**: ~68 个（矩阵-向量操作）
- **Level 3 BLAS**: ~54 个（矩阵-矩阵操作）
- **Batched operations**: ~48 个（批量操作）

### 操作类型分布（Top 15）

| 操作        | 函数数 | 说明                     |
|-----------|------|------------------------|
| gemm      | 5    | 通用矩阵乘法                 |
| gemmbatched | 5  | 批量矩阵乘法                 |
| gemmstridedbatched | 5 | 步进批量矩阵乘法       |
| scal      | 4    | 向量缩放                   |
| axpy      | 4    | y = alpha*x + y        |
| copy      | 4    | 向量复制                   |
| swap      | 4    | 向量交换                   |
| rot       | 4    | Givens 旋转            |
| gemv      | 4    | 矩阵-向量乘法              |
| trsv      | 4    | 三角矩阵求解（向量）         |
| symv      | 4    | 对称矩阵-向量乘法           |
| hemv      | 4    | Hermitian 矩阵-向量乘法  |
| ger       | 4    | 秩 1 更新                 |
| syr       | 4    | 对称秩 1 更新             |
| her       | 4    | Hermitian 秩 1 更新    |

## 检查生成结果

### 查看生成统计

```bash
# 查看生成成功率
cat output_ut_cublas/*/generation_summary.json | python -m json.tool | grep -E "total|successful|failed|success_rate"
```

### 快速验证生成的代码

```python
# 检查生成的文件数量
ls -1 output_ut_cublas/ut_*/ut_0/*.py | wc -l

# 检查某个文件的内容
cat output_ut_cublas/ut_*/ut_0/test_gemm_float32.py
```

### 运行生成的测试

```bash
# 运行单个测试
pytest output_ut_cublas/ut_*/ut_0/test_gemm_float32.py -v

# 运行所有测试
pytest output_ut_cublas/ut_*/ut_0/ -v
```

## 常见问题

### Q1: 生成失败怎么办？

检查 `generation_summary.json` 中的错误信息：

```bash
cat output_ut_cublas/*/generation_summary.json | python -c "
import sys, json
data = json.load(sys.stdin)
for r in data['results']:
    if not r['success']:
        print(f\"Failed: {r['cublas_function']}: {r.get('error', 'Unknown')}\")"
```

### Q2: 如何调试生成的 prompt？

使用 `--verbose` 标志查看详细日志：

```bash
python script/cublas_cupy/generate_ut_cublas.py \
    --name gemm \
    --verbose 2>&1 | tee generation.log
```

生成的 prompt 会保存在运行目录的日志中。

### Q3: 支持哪些数据类型？

- `float16` (H): 7 个函数
- `float32` (S): 51 个函数
- `float64` (D): 49 个函数
- `complex64` (C): 61 个函数
- `complex128` (Z): 52 个函数

### Q4: 如何只生成特定数据类型的测试？

目前脚本会为每个操作的所有数据类型变体生成测试。如果只想要 float32：

```bash
# 生成后手动过滤
python script/cublas_cupy/generate_ut_cublas.py --name gemm
find output_ut_cublas/ -name "*float32.py"
```

## 下一步工作

生成测试函数后，你可以：

1. **验证测试**: 运行 pytest 确保测试通过
2. **集成到 FlagBench**: 将测试整合到现有测试框架
3. **性能对比**: 使用生成的 baseline 对比 Triton kernel 性能
4. **修复失败的测试**: 根据错误信息调整 prompt 或手动修复

## 技术细节

### 工作流程

1. **加载 schema**: 从 `cublas_ops.json` 读取 239 个函数签名
2. **过滤函数**: 排除非 BLAS 函数（logger、handle 等）→ 219 个有效函数
3. **创建参数**: 为每个函数构造 `TestFuncGenerateArgs`
   - `kernel_name` = 操作名（如 "gemm"）
   - `use_cublas` = True（标记使用 cuBLAS prompt）
   - `cublas_schema` = 完整函数信息
4. **调用 LLM**: 通过 `CuBLASTestFuncGenerator` 调用 `generate_prompt_for_cublas()`
5. **保存结果**: 生成的代码保存到 `output_ut_cublas/`

### 动态属性机制

由于 `TestFuncGenerateArgs` 是 Pydantic BaseModel，不支持直接添加新字段。脚本通过 `__dict__` 动态添加属性：

```python
gen_arg = TestFuncGenerateArgs(kernel_name="gemm", operators={}, ...)
gen_arg.__dict__['use_cublas'] = True
gen_arg.__dict__['cublas_schema'] = {...}
```

`CuBLASTestFuncGenerator._init_data()` 会保留这些动态属性。

### Prompt 生成逻辑

调用链：
1. `generate_samples()` → 构造 `gen_args` 列表
2. `generator(gen_args)` → 调用 `CuBLASTestFuncGenerator`
3. `generate_prompt()` → 检测 `use_cublas=True`
4. `generate_prompt_for_cublas()` → 生成最终 prompt（在 `test_func_generator.py:510`）

## 参考资料

- cuBLAS 文档: https://docs.nvidia.com/cuda/cublas/
- CuPy cuBLAS 封装: https://docs.cupy.dev/en/stable/reference/cublas.html
- DLPack 互操作: https://github.com/dmlc/dlpack
- FlagBench 架构: `AGENTS.md`
