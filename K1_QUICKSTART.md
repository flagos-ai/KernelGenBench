# FlagBench K1 快速使用指南

## 环境设置

```bash
# 激活环境
source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench

# 进入项目目录
cd /share/project/zpy/flagbench
```

## 测试已有的三个算子

### 1. 测试 CuPy Baseline（PyTorch Custom Ops）

```bash
cd /tmp
python /share/project/zpy/flagbench/src/flagbench/ops/kaldi_ops.py
```

**预期输出**:
```
Testing Kaldi PyTorch Custom Ops
✓ Test 1 PASSED! (copy_low_upp)
✓ Test 2 PASSED! (add_mat)
✓ All PyTorch custom ops tests passed!
```

### 2. 测试 Triton Kernels

```bash
cd /tmp

# 测试 copy_low_upp
python /share/project/zpy/flagbench/triton_kernels_k1/copy_low_upp_kernel.py

# 测试 copy_upp_low
python /share/project/zpy/flagbench/triton_kernels_k1/copy_upp_low_kernel.py

# 测试 add_mat
python /share/project/zpy/flagbench/triton_kernels_k1/add_mat_kernel.py
```

### 3. 运行完整流程测试（正确性 + 性能）

```bash
cd /tmp
python /share/project/zpy/flagbench/test_k1_full_pipeline.py
```

**预期结果**:
- ✓ 所有正确性测试通过
- ✓ Triton 比 CuPy baseline 快 2.9-3.7x

## 在 Python 中使用

### 使用 CuPy Baseline

```python
import torch
from flagbench.ops import kaldi_ops  # 这会自动注册 torch.ops.kaldi.*

# copy_low_upp
A = torch.tril(torch.randn(64, 64, device='cuda'))
torch.ops.kaldi.copy_low_upp(A)  # 原位修改

# copy_upp_low
A = torch.triu(torch.randn(64, 64, device='cuda'))
torch.ops.kaldi.copy_upp_low(A)  # 原位修改

# add_mat
dst = torch.randn(128, 256, device='cuda')
src = torch.randn(128, 256, device='cuda')
torch.ops.kaldi.add_mat(dst, src, alpha=2.0)  # dst = 2.0 * src + dst
```

### 使用 Triton 实现

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path('/share/project/zpy/flagbench/triton_kernels_k1')))

import torch
from copy_low_upp_kernel import copy_low_upp
from copy_upp_low_kernel import copy_upp_low
from add_mat_kernel import add_mat

# 使用方式与 CuPy baseline 相同
A = torch.tril(torch.randn(64, 64, device='cuda'))
copy_low_upp(A)
```

## 为新算子添加支持

### 步骤 1: 更新 IMPL_INFO_K1

编辑 `src/flagbench/dataset/kernel_list_k1.py`，添加算子定义：

```python
"your_kernel_name": {
    "description": "算子功能描述",
    "input_args": [
        {"name": "arg1", "type": "torch.Tensor", "desc": "参数说明"},
        {"name": "arg2", "type": "float", "desc": "参数说明"}
    ],
    "output_args": [{"type": "None", "desc": "原位操作，返回None"}],
    "torch_op": "torch.ops.kaldi.your_kernel_name",
    "algorithm": "算法描述",
    "hints": "实现提示"
},
```

### 步骤 2: 实现 CuPy Baseline

在 `src/flagbench/ops/kaldi_ops.py` 中添加：

```python
@torch.library.custom_op("kaldi::your_kernel_name", mutates_args={"arg1"})
def your_kernel_name(arg1: torch.Tensor, arg2: float) -> None:
    """算子文档"""
    # 1. 输入验证
    assert arg1.is_cuda, "Tensor must be on CUDA device"
    
    # 2. 转换为 CuPy
    arg1_cp = _torch_to_cupy(arg1)
    
    # 3. 调用 CuPy kernel
    kernel = kaldi_lib.your_kernel_name
    kernel(arg1_cp, arg2, grid=..., block=..., dtype=...)

@your_kernel_name.register_fake
def _(arg1: torch.Tensor, arg2: float) -> None:
    pass
```

### 步骤 3: 编写 Triton Kernel

在 `triton_kernels_k1/` 目录创建 `your_kernel_name_kernel.py`：

```python
import torch
import triton
import triton.language as tl

@triton.jit
def your_kernel_name_kernel(...):
    """Triton kernel 实现"""
    pass

def your_kernel_name(...):
    """Python wrapper"""
    # 1. 输入验证
    # 2. 计算 grid/block
    # 3. 调用 kernel
    your_kernel_name_kernel[grid](...)
```

### 步骤 4: 测试

创建测试脚本验证正确性和性能。

## 常用命令速查

### 查看 IMPL_INFO_K1 中的所有算子

```python
from flagbench.dataset.kernel_list_k1 import IMPL_INFO_K1
print(f"Total kernels: {len(IMPL_INFO_K1)}")
print("Available kernels:", list(IMPL_INFO_K1.keys())[:10])
```

### 检查 CuPy 内核是否注册

```python
import sys
sys.path.insert(0, '/share/project/zpy/flagbench/script/cupy')
from kaldi_kernel_wrapper import kaldi_lib

print("Registered kernels:", list(kaldi_lib.kernels.keys()))
```

### 性能基准测试模板

```python
import time
import torch

def benchmark(func, *args, num_iters=1000):
    # Warmup
    for _ in range(10):
        func(*[a.clone() if isinstance(a, torch.Tensor) else a for a in args])
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iters):
        func(*[a.clone() if isinstance(a, torch.Tensor) else a for a in args])
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"Time: {elapsed:.3f}s ({elapsed/num_iters*1000:.3f}ms per iter)")
    return elapsed

# 使用
A = torch.randn(512, 512, device='cuda')
t1 = benchmark(torch.ops.kaldi.copy_low_upp, A)
t2 = benchmark(triton_copy_low_upp, A)
print(f"Speedup: {t1/t2:.2f}x")
```

## 故障排查

### 问题 1: 找不到 kaldi_ops 模块

**解决方案**:
```bash
export PYTHONPATH=/share/project/zpy/flagbench/src:$PYTHONPATH
```

### 问题 2: CuPy kernel 加载失败

**检查**:
```python
import sys
sys.path.insert(0, '/share/project/zpy/flagbench/script/cupy')
from kaldi_kernel_wrapper import kaldi_lib
print(kaldi_lib.copy_low_upp)  # 应该显示 kernel 对象
```

### 问题 3: Triton 编译错误

**常见原因**:
- 张量不是 contiguous：添加 `.contiguous()`
- 张量不在 CUDA 上：检查 `.is_cuda`
- grid/block 配置错误：检查维度计算

### 问题 4: 精度不匹配

**调试**:
```python
import torch
A_baseline = ...
A_triton = A_baseline.clone()

torch.ops.kaldi.your_kernel(A_baseline)
triton_your_kernel(A_triton)

diff = torch.abs(A_baseline - A_triton)
print(f"Max diff: {diff.max():.2e}")
print(f"Mean diff: {diff.mean():.2e}")
print(f"Locations of large diffs:")
print(torch.where(diff > 1e-5))
```

## 文件位置速查

| 类型 | 位置 |
|------|------|
| IMPL_INFO_K1 定义 | `src/flagbench/dataset/kernel_list_k1.py` |
| CuPy Baseline | `src/flagbench/ops/kaldi_ops.py` |
| CuPy Wrapper (底层) | `script/cupy/kaldi_kernel_wrapper.py` |
| Triton Kernels | `triton_kernels_k1/*.py` |
| 完整测试 | `test_k1_full_pipeline.py` |
| 生成脚本 | `script/generate_*_sample4k1.py` |

## 下一步

1. 选择更多 K1 算子进行集成
2. 使用 LLM 自动生成 Triton 实现
3. 建立完整的 benchmark 数据库

## 相关文档

- 完整报告: `K1_INTEGRATION_REPORT.md`
- 项目指南: `AGENTS.md`
- CuPy 演示: `script/cupy/demo_kaldi_kernels.py`
