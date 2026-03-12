# CuPy cuBLAS Baseline 方案

## 项目目标

为 FlagBench 构建基于 CuPy 的 cuBLAS baseline，用于对比 Triton kernel 性能。

---

## 📊 当前进度：第一步完成

### ✅ 已完成：cuBLAS Schema 自动提取

**时间：** 2026-01-16  
**状态：** ✅ 完成并通过人工验证

#### 成果

从 cuBLAS 头文件自动提取了 **239 个函数**的完整签名，包括：
- ✅ Level 1 BLAS (向量操作): 34 函数
- ✅ Level 2 BLAS (矩阵-向量): 68 函数
- ✅ Level 3 BLAS (矩阵-矩阵): 54 函数
- ✅ Batched 操作: 48 函数
- ✅ FP16/FP32/FP64/Complex 全覆盖

#### 文件

| 文件 | 描述 | 行数 |
|------|------|------|
| `parse_cublas_schema.py` | 自动解析脚本 | 355 行 |
| `cublas_ops.json` | 生成的 schema | 239 函数 |

#### 运行命令

```bash
# 激活环境
source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench

# 生成 schema
cd /share/project/zpy/flagbench
python script/cublas_cupy/parse_cublas_schema.py
```

#### Schema 格式

```json
{
  "version": "1.0",
  "source": "cublas_api.h",
  "total_functions": 239,
  "functions": [
    {
      "name": "cublasSgemm_v2",
      "operation": "gemm",
      "dtype": "float32",
      "args": [
        {"name": "handle", "type": "cublasHandle_t", "role": "context"},
        {"name": "transa", "type": "cublasOperation_t", "role": "value"},
        {"name": "m", "type": "int", "role": "value"},
        {"name": "alpha", "type": "const float*", "role": "scalar"},
        {"name": "A", "type": "const float*", "role": "input"},
        {"name": "C", "type": "float*", "role": "inout"}
      ]
    }
  ]
}
```

#### 参数角色说明

| 角色 | 描述 | 示例 |
|------|------|------|
| `context` | cuBLAS handle | `cublasHandle_t handle` |
| `scalar` | 标量参数（alpha, beta） | `const float* alpha` |
| `input` | 只读输入（const 指针） | `const float* A` |
| `output` | 只写输出（result 参数） | `float* result` |
| `inout` | 输入输出（in-place） | `float* C` (GEMM), `float* y` (AXPY) |
| `value` | 维度参数（非指针） | `int m, int n, int k` |

#### 数据类型推断

| 函数名前缀 | 数据类型 | 示例 |
|-----------|---------|------|
| `cublasS*` | `float32` | `cublasSgemm_v2` |
| `cublasD*` | `float64` | `cublasDgemm_v2` |
| `cublasH*` | `float16` | `cublasHgemm` |
| `cublasC*` | `complex64` | `cublasCgemm_v2` |
| `cublasZ*` | `complex128` | `cublasZgemm_v2` |

#### 人工验证

验证了 4 个关键函数，全部通过：

```bash
✅ cublasSgemm_v2: C 参数 = inout (正确)
✅ cublasSaxpy_v2: y 参数 = inout (正确)
✅ cublasSdot_v2: result 参数 = output (正确)
✅ cublasSscal_v2: x 参数 = inout (正确)
```

---

## 🚀 下一步工作

### ✅ 第二步：Prompt 生成器（已完成）

**位置：** `/share/project/zpy/flagbench/src/generator/test_func_generator.py`

已在 `TestFuncGenerator` 类中添加了 `generate_prompt_for_cublas()` 方法，可以基于 cuBLAS schema 自动生成 LLM prompt。

**使用方法：**

```bash
# 简单示例（直接运行）
cd /share/project/zpy/flagbench
python script/cublas_cupy/simple_example.py
```

这会为 `cublasSgemm_v2` 生成 prompt 并保存到 `/tmp/cublasSgemm_v2_prompt.txt`。

**生成的 Prompt 特点：**
- ✅ 自动读取 cuBLAS schema
- ✅ 区分参数角色（input/output/inout/scalar/value）
- ✅ 提供 DLPack 转换示例
- ✅ 包含完整的函数模板
- ✅ 针对不同操作类型（gemm, axpy, dot 等）提供专门示例

**文件：**
- `src/generator/test_func_generator.py:generate_prompt_for_cublas()` - Prompt 生成函数（200+ 行）
- `script/cublas_cupy/simple_example.py` - 简单使用示例
- `script/cublas_cupy/example_generate_prompt.py` - 完整示例（集成到 FlagBench）

---

### ✅ 第三步：批量测试函数生成脚本（已完成）

**时间：** 2026-01-19  
**状态：** ✅ 完成并测试通过

**位置：** `/share/project/zpy/flagbench/script/cublas_cupy/generate_ut_cublas.py`

创建了类似 `generate_ut_sample4k1.py` 的批量生成脚本，可以为所有 cuBLAS 函数自动生成测试代码。

**核心功能：**
- ✅ 从 `cublas_ops.json` 加载 239 个 cuBLAS 函数
- ✅ 自动过滤非 BLAS 函数（logger、handle 等）→ 219 个有效函数
- ✅ 调用 LLM 批量生成测试函数（CuPy baseline vs Triton）
- ✅ 支持单个/多个/全部操作生成
- ✅ 自动重试机制（处理 API 失败）
- ✅ 并行生成（可配置 workers）
- ✅ 生成统计报告（JSON 格式）

**使用方法：**

```bash
# 1. 激活环境（必须！）
source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench

# 2. 为单个操作生成（推荐先测试）
cd /share/project/zpy/flagbench
python script/cublas_cupy/generate_ut_cublas.py --name gemm

# 3. 批量生成常用操作
for op in gemm axpy dot scal copy; do
    python script/cublas_cupy/generate_ut_cublas.py --name $op
done

# 4. 生成所有 219 个有效函数（⚠️ 会调用 LLM 219 次）
python script/cublas_cupy/generate_ut_cublas.py --name all

# 5. 自定义参数
python script/cublas_cupy/generate_ut_cublas.py \
    --name gemm \
    --model-name deepseek-v3-0324 \
    --num-workers 20 \
    --max-retries 5
```

**输出目录：**
```
output_ut_cublas/
└── ut_cublas_{model}_{timestamp}/
    ├── ut_0/
    │   ├── test_gemm_float32.py
    │   ├── test_gemm_float64.py
    │   └── ...
    └── generation_summary.json
```

**详细文档：**
- **使用指南：** `script/cublas_cupy/GENERATE_UT_README.md` - 详细使用说明、参数配置、FAQ
- **技术实现：** `script/cublas_cupy/generate_ut_cublas.py` - 主脚本（约 500 行）

**统计信息：**
- 总函数数：239（来自 cublas_v2.h）
- 有效 BLAS 函数：219（已过滤）
- 主要操作类型：gemm (5个变体), axpy (4个), dot (4个), scal (4个), 等

---

### 第四步：端到端测试（待做）

**目标：** 运行生成的测试函数，验证 CuPy baseline 正确性

**步骤：**
1. 运行 `generate_ut_cublas.py` 生成测试
2. 执行 pytest 验证测试通过
3. 性能对比 Triton kernel vs cuBLAS

---

## 🔍 生成的测试函数格式示例

**示例输出：**
```python
import torch
import cupy as cp
from torch.utils.dlpack import to_dlpack, from_dlpack

@label("gemm")
@parametrize("M, N, K", [(128, 256, 64), (512, 512, 256)])
@parametrize("dtype", [torch.float32])
def test_gemm_float32(M, N, K, dtype):
    # Initialize inputs
    A = torch.randn(M, K, dtype=dtype, device='cuda')
    B = torch.randn(K, N, dtype=dtype, device='cuda')
    
    # CuPy baseline (cuBLAS reference)
    A_cp = cp.from_dlpack(to_dlpack(A))
    B_cp = cp.from_dlpack(to_dlpack(B))
    ref_out_cp = cp.dot(A_cp, B_cp)
    ref_out = from_dlpack(ref_out_cp.toDlpack())
    
    # Triton implementation
    with flagbench.use_gems(REGISTERED_OPS):
        act_out = torch.ops.aten.mm(A, B)
    
    # Compare
    assert_close(act_out, ref_out, dtype=dtype)
```

**注意：** 实际生成的代码由 LLM 生成，格式可能略有不同，但逻辑相同。

---

### 第五步：集成到 FlagBench（待做）

**目标：** 将 cuBLAS baseline 整合到 FlagBench 主流程

**修改点：**
1. ✅ 测试函数生成 prompt（已完成 - `generate_prompt_for_cublas()`）
2. 自动发现 cuBLAS baseline（根据算子名匹配）
3. 性能对比报告生成

---

## 📖 快速开始指南

### 新手上手（5 分钟）

```bash
# 1. 激活环境
source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench

# 2. 进入项目目录
cd /share/project/zpy/flagbench

# 3. 为 GEMM 操作生成测试（单个操作测试）
python script/cublas_cupy/generate_ut_cublas.py --name gemm

# 4. 查看生成结果
ls -lh output_ut_cublas/ut_cublas_*/ut_0/

# 5. 查看生成统计
cat output_ut_cublas/ut_cublas_*/generation_summary.json | python -m json.tool | head -20

# 6. 运行生成的测试（需要 Triton kernel 实现）
pytest output_ut_cublas/ut_cublas_*/ut_0/test_gemm_*.py -v
```

### 批量生成常用操作

```bash
# 生成 Level 1/2/3 BLAS 核心操作（约 20 个）
for op in gemm axpy dot scal copy swap gemv ger symv syr; do
    echo "Generating $op..."
    python script/cublas_cupy/generate_ut_cublas.py --name $op
done
```

### 生成全部操作（慎用）

```bash
# ⚠️ 会调用 LLM 219 次，可能需要 30-60 分钟和显著费用
python script/cublas_cupy/generate_ut_cublas.py \
    --name all \
    --num-workers 20 \
    --max-retries 5
```

---

## 📁 文件结构（更新）

```
/share/project/zpy/flagbench/script/cublas_cupy/
├── parse_cublas_schema.py        # Schema 解析脚本（355 行）
├── cublas_ops.json                # 生成的 schema（239 函数）
├── generate_ut_cublas.py          # ✨ 批量测试生成脚本（500 行）
├── simple_example.py              # Prompt 生成示例
├── example_generate_prompt.py     # 完整集成示例
├── README.md                      # 本文件（项目概览）
├── GENERATE_UT_README.md          # ✨ 详细使用指南
└── USAGE.md                       # Prompt 生成器使用说明
```

**修改的文件：**
- `src/generator/test_func_generator.py` - 添加 `generate_prompt_for_cublas()` 方法

**输出目录：**
- `output_ut_cublas/` - 生成的测试函数（自动创建）

---

## 📚 技术文档

### 为什么选择 CuPy？

#### 优势

1. **零编译成本**
   - 不需要写 C++ wrapper
   - 不需要管理 CUDA 头文件
   - 不需要处理 PyTorch extension 编译

2. **API 完整**
   - CuPy 完整封装 cuBLAS/cuFFT/cuSPARSE
   - 接口稳定，文档完善
   - 社区维护，bug 少

3. **与 PyTorch 互操作性好**
   - DLPack 零拷贝转换
   - 性能开销 < 1μs（仅元数据映射）

4. **代码简洁**
   - 几行代码搞定 baseline
   - 无需维护复杂的 C++ 代码

#### 性能

- **DLPack 转换开销：** ~1μs（零拷贝）
- **cuBLAS 调用：** 与 C++ 完全相同（CuPy 是薄封装）
- **总开销：** < 0.1%（对于计算密集型算子）

#### Row-Major vs Column-Major

**问题：** PyTorch 是 Row-Major，cuBLAS 是 Column-Major

**解决方案：**
- **选项 A（推荐）：** 使用 `cupy.dot()` 高层 API，自动处理转换
- **选项 B：** 使用 `cupy.cuda.cublas.*` 底层 API，手动交换维度（和 C++ 一样）

对于 baseline，选项 A 足够。

---

## 🔧 技术细节

### Schema 解析器实现

#### 核心算法

1. **正则表达式匹配函数声明**
   ```python
   pattern = r'(?:CUBLASAPI\s+)?cublasStatus_t\s+(?:CUBLASWINAPI\s+)?(cublas[A-Z][a-zA-Z0-9_]+)\s*\(([^;]+?)\)\s*;'
   ```

2. **智能参数分割**
   - 处理函数指针（括号嵌套）
   - 处理数组参数（方括号）
   - 按逗号分割，但保留嵌套内容

3. **数据类型推断**
   ```python
   func_name[6] == 'S' → float32
   func_name[6] == 'D' → float64
   func_name[6] == 'H' → float16
   ```

4. **参数角色分类**
   - 优先级：handle > value (非指针) > scalar (alpha/beta) > const 指针 > 非 const 指针
   - 特殊处理：result/info/work 参数

### 已知限制

1. **Extended 函数（_Ex）的 dtype 为 unknown**
   - 原因：使用 `void*` + `cudaDataType` 动态指定类型
   - 影响：19 个函数
   - 解决方案：手动标注或跳过

2. **64 位版本（_64）被过滤**
   - 原因：只需要标准版本
   - 影响：约 100 个函数被排除

3. **某些特殊函数未提取**
   - 例如：Logger 相关函数（但也提取了，dtype=unknown）

---

## 🎯 预期成果

完成后，FlagBench 将能够：

1. **自动生成 cuBLAS baseline**
   - 输入：算子名（如 `sgemm`）
   - 输出：CuPy wrapper 函数

2. **对比性能**
   - Triton kernel vs cuBLAS (via CuPy)
   - 自动生成性能报告

3. **零维护成本**
   - Schema 从头文件自动提取
   - CuPy 封装稳定，无需更新

---

## 📁 文件结构

```
/share/project/zpy/flagbench/script/cublas_cupy/
├── parse_cublas_schema.py   # Schema 解析脚本
├── cublas_ops.json           # 生成的 schema (239 函数)
├── README.md                 # 本文件
└── (待添加)
    ├── generate_cupy_wrapper.py   # 生成 CuPy wrapper (第三步)
    ├── test_cupy_baseline.py      # 测试脚本 (第五步)
    └── cupy_baselines/            # 生成的 baseline 函数
```

---

## 🔗 相关资源

- [CuPy 官方文档](https://docs.cupy.dev/)
- [CuPy cuBLAS API](https://docs.cupy.dev/en/stable/reference/cublas.html)
- [cuBLAS 官方文档](https://docs.nvidia.com/cuda/cublas/)
- [DLPack 协议](https://github.com/dmlc/dlpack)
