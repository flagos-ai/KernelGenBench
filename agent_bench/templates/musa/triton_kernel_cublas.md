# Triton Kernel 实现任务 (cuBLAS)（摩尔线程 MUSA）

你需要为 cuBLAS baseline 函数实现一个在摩尔线程 MUSA 设备上运行的 Triton kernel。

## 任务信息

- **算子名称**: {{OPERATOR}}
- **完整名称**: {{FULL_NAME}}
- **GPU ID**: {{GPU_ID}}

## 运行环境

- **硬件**: 摩尔线程 MUSA 设备
- **软件栈**: torch_musa（基于 PyTorch），Triton for MUSA
- 所有涉及 MUSA 的命令必须加上 `MUSA_VISIBLE_DEVICES={{GPU_ID}}` 前缀
- Python 路径: `{{PYTHON_PATH}}`

## MUSA 设备注意事项（必须遵守）

- **`import torch` 后必须紧跟 `import torch_musa`**，否则 musa 设备不可用
- 设备类型是 `musa`，所有 device 相关 API 使用 `musa`，例如：
  - `device = torch.device("musa:0")`
  - `torch.musa.synchronize()`
  - `tensor.to('musa')`
- 环境变量使用 `MUSA_VISIBLE_DEVICES` 而非 `CUDA_VISIBLE_DEVICES`
- Triton kernel 的编写方式与 NVIDIA GPU 基本一致，但需注意：
  - 某些高级 Triton 特性可能不支持，优先使用基础 Triton 操作
  - 避免依赖 CUDA 特有的硬件特性

## Baseline 函数

以下是你需要用 Triton 实现的 cuBLAS baseline 函数（通过 ctypes 调用 cuBLAS C API）：

```python
{{BASELINE_CODE}}
```

### 函数签名

{{OP_SIGNATURES}}

### 输入输出参数

{{INPUT_ARGS}}

## 实现要求

### 1. 代码结构

你的实现必须包含：
1. **Triton kernel 函数**：使用 `@triton.jit` 装饰器定义核心计算逻辑
2. **Python wrapper 函数**：与 baseline 函数**完全相同的签名**（函数名、参数名、参数顺序必须一致）

### 2. 关键要求

- wrapper 函数签名必须与 baseline **完全一致**，否则测试会直接失败
- 对于 float16/bfloat16 输入，内部计算使用 float32 累加
- 矩阵运算使用 `allow_tf32=False` 保持精度
- 正确处理 BLAS 参数（如 `incx`、`incy`、`lda`、`ldb`、`ldc`、`trans` 等）
- 注意 cuBLAS 的列优先 (column-major) 存储约定

### 3. 测试环境

你的实现将按如下方式测试：
```python
# Baseline (cuBLAS C API wrapper)
from kernelgenbench.dataset.baseline.cublas.{{OPERATOR}} import {{OPERATOR}} as baseline_{{OPERATOR}}
ref_out = baseline_{{OPERATOR}}(...)

# Your Triton implementation
act_out = your_{{OPERATOR}}(...)

# Accuracy verification
assert_close(act_out, ref_out, dtype)
```

## 示例

以下是 `saxpy` baseline 函数及其对应的 Triton kernel 实现：

**cuBLAS baseline 函数：**
```python
def saxpy(n, alpha, x, incx, y, incy):
    '''SAXPY: y = alpha * x + y'''
    # cuBLAS C API call via ctypes
    cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy)
```

**Triton kernel 实现：**
```python
import torch
import torch_musa
import triton
import triton.language as tl

@triton.jit
def saxpy_kernel(n, alpha, x_ptr, incx, y_ptr, incy, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n

    x_idx = offsets * incx
    y_idx = offsets * incy

    x = tl.load(x_ptr + x_idx, mask=mask)
    y = tl.load(y_ptr + y_idx, mask=mask)

    result = alpha * x + y
    tl.store(y_ptr + y_idx, result, mask=mask)

def saxpy(n, alpha, x, incx, y, incy):
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    saxpy_kernel[grid](n, alpha, x, incx, y, incy, BLOCK_SIZE=1024)
    return y
```

## IMPORTANT - No Cheating

- You MUST implement the algorithm using Triton kernels (`@triton.jit`)
- Do NOT call the baseline function or cuBLAS C API directly
- Do NOT use ctypes to call cuBLAS functions
- Do NOT use `torch.ops` to call the original operator
- Your implementation must be a pure Triton kernel solution

## 输出要求

**重要**：请直接在回复中输出完整的 Python 代码，要求：

1. 代码必须用 ```python ... ``` 代码块包裹
2. 代码可以直接运行，无需修改
3. **必须包含 `import torch` 和 `import torch_musa`**
4. 包含 wrapper 函数，签名与 baseline **完全一致**
5. 不要包含测试代码或 benchmark 代码
6. 不要添加额外的解释文字，只输出代码块
7. **不要将代码写入文件**，直接在回复中输出即可

示例输出格式：
```python
import torch
import torch_musa
import triton
import triton.language as tl

# 你的实现代码...
```

{{REFERENCE_CODE}}
