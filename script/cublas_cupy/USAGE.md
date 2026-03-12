# cuBLAS Prompt 生成器使用指南

## 快速开始

### 1. 运行简单示例

```bash
cd /share/project/zpy/flagbench
python script/cublas_cupy/simple_example.py
```

输出会显示为 `cublasSgemm_v2` 生成的 prompt，并保存到 `/tmp/cublasSgemm_v2_prompt.txt`。

### 2. 查看生成的 Prompt

```bash
cat /tmp/cublasSgemm_v2_prompt.txt
```

你会看到类似这样的内容：

```
# 任务：为 cublasSgemm_v2 生成 CuPy baseline 函数

## cuBLAS 函数信息
- 函数名: cublasSgemm_v2
- 操作类型: gemm
- 数据类型: float32

## 参数列表:
- handle (context): cublasHandle_t
- transa (value): cublasOperation_t
- transb (value): cublasOperation_t
- m (value): int
- n (value): int
- k (value): int
- alpha (scalar): const float*
- A (input): const float*
- lda (value): int
- B (input): const float*
- ldb (value): int
- beta (scalar): const float*
- C (inout): float*
- ldc (value): int

## 要求
生成一个 Python 函数 `gemm_cublas_baseline`，实现以下功能：
1. 接收 PyTorch tensor 作为输入
2. 使用 DLPack 转换为 CuPy array（零拷贝）
3. 调用 CuPy 的高层 API（如 cp.dot()）
4. 将结果转回 PyTorch tensor

## 示例代码框架：
...
```

### 3. 将 Prompt 发送给 LLM

你可以：
- 复制 prompt 内容，发送给 ChatGPT / Claude
- 或者使用 FlagBench 的 LLM API 自动生成

### 4. 保存生成的函数

LLM 会生成类似这样的代码：

```python
def gemm_cublas_baseline(A, B, C=None, alpha=1.0, beta=0.0, transa=False, transb=False):
    """CuPy baseline for cublasSgemm_v2"""
    import cupy as cp
    from torch.utils.dlpack import to_dlpack, from_dlpack
    
    A_cp = cp.from_dlpack(to_dlpack(A))
    B_cp = cp.from_dlpack(to_dlpack(B))
    
    if transa:
        A_cp = A_cp.T
    if transb:
        B_cp = B_cp.T
    
    if C is None:
        result = alpha * cp.dot(A_cp, B_cp)
    else:
        C_cp = cp.from_dlpack(to_dlpack(C))
        result = alpha * cp.dot(A_cp, B_cp) + beta * C_cp
    
    return from_dlpack(result.toDlpack())
```

保存到 `script/cublas_cupy/baselines/gemm_baseline.py`

---

## 为其他算子生成 Prompt

修改 `simple_example.py` 中的 `functions_to_test` 列表：

```python
functions_to_test = [
    "cublasSgemm_v2",    # GEMM
    "cublasSaxpy_v2",    # AXPY
    "cublasSdot_v2",     # DOT
    "cublasSscal_v2",    # SCAL
    "cublasDgemm_v2",    # DGEMM (FP64)
    "cublasHgemm",       # HGEMM (FP16)
]
```

---

## 集成到 FlagBench 流程（高级）

### 使用完整的 TestFuncGenerator

```python
import sys
sys.path.insert(0, 'src')

from generator.test_func_generator import TestFuncGenerator
from generator.sampler.generate_samples import TestFuncGenerateArgs
import json

# 加载 schema
with open('script/cublas_cupy/cublas_ops.json') as f:
    data = json.load(f)

# 找到 cublasSgemm_v2
schema = next(f for f in data['functions'] if f['name'] == 'cublasSgemm_v2')

# 创建生成器
class DummyConfig:
    pass

generator = TestFuncGenerator(DummyConfig())

# 创建参数
info = TestFuncGenerateArgs(
    kernel_name="matmul_triton",
    ops_namespace="aten",
    op_name="test_matmul",
    sample_id=0
)

# 生成 prompt
prompt = generator.generate_prompt_for_cublas(info, schema)

print(prompt)
```

---

## 文件结构

```
/share/project/zpy/flagbench/
├── script/cublas_cupy/
│   ├── cublas_ops.json                # cuBLAS schema (239 函数)
│   ├── parse_cublas_schema.py         # Schema 解析器
│   ├── simple_example.py              # 简单示例 ✅ 推荐新手使用
│   ├── example_generate_prompt.py     # 完整示例
│   ├── USAGE.md                       # 本文件
│   └── README.md                      # 项目总体说明
│
└── src/generator/
    └── test_func_generator.py
        └── generate_prompt_for_cublas()  # Prompt 生成函数 (200+ 行)
```

---

## 常见问题

### Q1: 我如何知道有哪些 cuBLAS 函数可用？

查看 `cublas_ops.json`：

```bash
cat script/cublas_cupy/cublas_ops.json | python -c "
import sys, json
data = json.load(sys.stdin)
print(f'Total: {len(data[\"functions\"])} functions')
print('\\nGEMM variants:')
for f in data['functions']:
    if f['operation'] == 'gemm':
        print(f'  - {f[\"name\"]} ({f[\"dtype\"]})')
"
```

### Q2: 生成的函数是否需要人工修改？

通常不需要，但建议：
1. 检查数据类型转换是否正确
2. 验证 in-place 操作的处理
3. 添加必要的参数检查

### Q3: 性能开销如何？

- DLPack 转换：~1μs（零拷贝）
- CuPy 调用：与直接调用 cuBLAS 相同
- 总开销：< 0.1%（对于 GEMM 等计算密集型算子）

### Q4: 为什么不直接用 PyTorch 的 cuBLAS bindings？

PyTorch 没有直接暴露 cuBLAS API。使用 CuPy 的优势：
1. 完整的 cuBLAS API 覆盖
2. 零拷贝转换（DLPack）
3. 简洁的 Python 接口

---

## 下一步

1. ✅ **已完成：** Schema 解析（239 个函数）
2. ✅ **已完成：** Prompt 生成器
3. **待做：** 批量生成所有算子的 baseline 函数
4. **待做：** 集成到 FlagBench 测试流程
5. **待做：** 性能对比 Triton vs cuBLAS

---

## 贡献者

- 初始实现：2026-01-19
- Schema 解析器：`parse_cublas_schema.py` (355 行)
- Prompt 生成器：`test_func_generator.py:generate_prompt_for_cublas()` (200+ 行)
