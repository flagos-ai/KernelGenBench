# Triton Kernel 实现任务

你需要为 PyTorch 算子实现一个 Triton kernel。

## 任务信息

- **算子名称**: {{OPERATOR}}
- **完整名称**: {{FULL_NAME}}
- **GPU ID**: {{GPU_ID}}

## 运行环境

- 所有涉及 GPU 的命令必须加上 `CUDA_VISIBLE_DEVICES={{GPU_ID}}` 前缀
- Python 路径: `{{PYTHON_PATH}}`

## 算子规范

### 函数签名

{{OP_SIGNATURES}}

### 需要实现的接口

{{IMPL_INFO}}

### 输入输出参数

{{INPUT_ARGS}}

## 实现要求

### 1. 代码结构

你的实现必须包含：
1. **Triton kernel 函数**：使用 `@triton.jit` 装饰器定义核心计算逻辑
2. **Python wrapper 函数**：为每个 ATen 接口提供对应的 Python 函数

### 2. 关键要求

**必须处理：**
- **Broadcasting**: 支持不同 shape 的输入按 PyTorch 广播语义计算
- **Non-contiguous tensors**: 不要假设输入 tensor 是连续的，使用正确的 stride 计算
- **所有 overload 变体**: 必须实现所有列出的接口变体

**命名规范：**
- wrapper 函数名必须与 ATen 算子名一致
- 将 `.` 替换为 `_`（如 `add.Tensor` → `add_Tensor`）

### 3. 精度要求

- 对于 float16/bfloat16 输入，内部计算使用 float32 累加
- 矩阵运算使用 `allow_tf32=False` 保持精度

## 示例

以下是 `add` 算子的实现示例：

```python
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def add_Tensor(self: torch.Tensor, other: torch.Tensor, alpha: float = 1) -> torch.Tensor:
    """实现 aten::add.Tensor"""
    # 处理 broadcasting
    self, other = torch.broadcast_tensors(self, other)
    # 确保连续性
    self = self.contiguous()
    other = other.contiguous()

    output = torch.empty_like(self)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # 处理 alpha
    if alpha != 1:
        other = other * alpha

    add_kernel[grid](self, other, output, n_elements, BLOCK_SIZE=1024)

    return output


def add_Scalar(self: torch.Tensor, other: float, alpha: float = 1) -> torch.Tensor:
    """实现 aten::add.Scalar"""
    return add_Tensor(self, torch.full_like(self, other), alpha)


def add_out(self: torch.Tensor, other: torch.Tensor, alpha: float = 1, *, out: torch.Tensor) -> torch.Tensor:
    """实现 aten::add.out"""
    result = add_Tensor(self, other, alpha)
    out.copy_(result)
    return out
```

## 输出要求

**重要**：请直接在回复中输出完整的 Python 代码，要求：

1. 代码必须用 ```python ... ``` 代码块包裹
2. 代码可以直接运行，无需修改
3. 不要包含测试代码或 benchmark 代码
4. 不要添加额外的解释文字，只输出代码块
5. **不要将代码写入文件**，直接在回复中输出即可

示例输出格式：
```python
import torch
import triton
import triton.language as tl

# 你的实现代码...
```

{{REFERENCE_CODE}}
