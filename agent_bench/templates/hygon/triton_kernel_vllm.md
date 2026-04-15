# Triton Kernel 实现任务 (vLLM)（海光 DCU）

你需要为 vLLM baseline 函数实现一个在海光 DCU（Deep Computing Unit）上运行的 Triton kernel。

## 任务信息

- **算子名称**: {{OPERATOR}}
- **完整名称**: {{FULL_NAME}}
- **GPU ID**: {{GPU_ID}}

## 运行环境

- **硬件**: 海光 DCU（Deep Computing Unit）
- 所有涉及 GPU 的命令必须加上 `HIP_VISIBLE_DEVICES={{GPU_ID}}` 前缀
- Python 路径: `{{PYTHON_PATH}}`

## 海光 DCU 注意事项（必须遵守）

- 设备类型是 `cuda`，使用标准 PyTorch CUDA API（底层走 HIP），例如：
  - `device = torch.device("cuda:0")`
  - `torch.cuda.synchronize()`
  - `tensor.to('cuda')`
- 环境变量使用 `HIP_VISIBLE_DEVICES`
- **不需要额外的 import**，直接 `import torch` 即可
- Triton kernel 的编写方式与 NVIDIA GPU 基本一致，但需注意：
  - 海光 DCU 基于 ROCm/HIP 生态，提供 CUDA 兼容接口，但底层硬件架构不同
  - 某些高级 CUDA/Triton 特性可能不支持或行为不同，优先使用基础 Triton 操作
  - 避免依赖 NVIDIA 特有的硬件特性（如 Tensor Core 特定指令、CUDA 特有 intrinsics）
  - `tl.dot` 建议使用 `allow_tf32=False`（TF32 是 NVIDIA 特有功能）

## Baseline 函数

以下是你需要用 Triton 实现的 vLLM baseline 函数：

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
- 正确处理边界条件和 edge cases
- 注意 in-place 操作：如果 baseline 修改了输入参数（如 `out.copy_(...)`)，你的实现也必须如此

### 3. 测试环境

你的实现将按如下方式测试：
```python
# Baseline
from flagbench.dataset.baseline.vllm.{{OPERATOR}} import {{OPERATOR}} as baseline_{{OPERATOR}}
ref_out = baseline_{{OPERATOR}}(...)

# Your Triton implementation
import flagbench
act_out = flagbench.triton.{{OPERATOR}}(...)

# Accuracy verification
assert_close(act_out, ref_out, dtype)
```

## 示例

以下是 `rms_norm` baseline 函数及其对应的 Triton kernel 实现：

**vLLM baseline 函数：**
```python
def rms_norm_baseline(out, input, weight, epsilon):
    '''RMS normalization'''
    variance = input.pow(2).mean(-1, keepdim=True)
    input_normalized = input * torch.rsqrt(variance + epsilon)
    out.copy_(input_normalized * weight)
```

**Triton kernel 实现（包含 kernel 和 wrapper 函数）：**
```python
import torch
import triton
import triton.language as tl

@triton.jit
def _rms_norm_kernel(
    output_ptr, input_ptr, weight_ptr,
    n_cols, epsilon,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    input_ptrs = input_ptr + row_idx * n_cols + col_offsets
    input_row = tl.load(input_ptrs, mask=mask, other=0.0)

    variance = tl.sum(input_row * input_row, axis=0) / n_cols
    rstd = 1 / tl.sqrt(variance + epsilon)

    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0)
    output = input_row * rstd * weight

    output_ptrs = output_ptr + row_idx * n_cols + col_offsets
    tl.store(output_ptrs, output, mask=mask)

# Wrapper function with EXACT SAME signature as baseline
def rms_norm_baseline(out, input, weight, epsilon):
    n_rows, n_cols = input.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    grid = (n_rows,)
    _rms_norm_kernel[grid](out, input, weight, n_cols, epsilon, BLOCK_SIZE=BLOCK_SIZE)
```

## IMPORTANT - No Cheating

- You MUST implement the algorithm using Triton kernels (`@triton.jit`)
- Do NOT call the baseline function directly
- Do NOT import or use `vllm._custom_ops`, `_custom_ops`, or any CUDA C++ extensions
- Do NOT use `torch.ops` to call the original operator
- Your implementation must be a pure Triton kernel solution

## 输出要求

**重要**：请直接在回复中输出完整的 Python 代码，要求：

1. 代码必须用 ```python ... ``` 代码块包裹
2. 代码可以直接运行，无需修改
3. 包含 wrapper 函数，签名与 baseline **完全一致**
4. 不要包含测试代码或 benchmark 代码
5. 不要添加额外的解释文字，只输出代码块
6. **不要将代码写入文件**，直接在回复中输出即可

示例输出格式：
```python
import torch
import triton
import triton.language as tl

# 你的实现代码...
```

{{REFERENCE_CODE}}
