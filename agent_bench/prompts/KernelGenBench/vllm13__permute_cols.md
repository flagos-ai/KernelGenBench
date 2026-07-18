<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Triton Kernel Implementation Task

You need to implement a Triton kernel for a PyTorch operator.

## Task Information

- **Operator**: permute_cols
- **Full Name**: vllm13::permute_cols
- **GPU ID**: 0

## Environment

- All GPU commands must be prefixed with `CUDA_VISIBLE_DEVICES=0`
- Python path: `python`

## Operator Specification

### Function Signatures

- `permute_cols`: (See vllm13 documentation)

### Interfaces to Implement

- `permute_cols`

### Input/Output Arguments

(See vllm13 documentation for parameter details)

## Implementation Requirements

### 1. Code Structure

Your implementation must include:
1. **Triton kernel function**: core computation logic decorated with `@triton.jit`
2. **Python wrapper functions**: one for each ATen interface variant

### 2. Key Requirements

**Must handle:**
- **Broadcasting**: support inputs with different shapes following PyTorch broadcast semantics
- **Non-contiguous tensors**: do not assume inputs are contiguous, use correct stride calculations
- **All overload variants**: implement every listed interface variant

**Naming convention:**
- Wrapper function names must match ATen operator names
- Replace `.` with `_` (e.g. `add.Tensor` → `add_Tensor`)

### 3. Precision Requirements

- For float16/bfloat16 inputs, accumulate internally in float32
- Use `allow_tf32=False` for matrix operations to maintain precision

## Example

Implementation example for the `add` operator:

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
    """Implements aten::add.Tensor"""
    # Handle broadcasting
    self, other = torch.broadcast_tensors(self, other)
    # Ensure contiguity
    self = self.contiguous()
    other = other.contiguous()

    output = torch.empty_like(self)
    n_elements = output.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Handle alpha
    if alpha != 1:
        other = other * alpha

    add_kernel[grid](self, other, output, n_elements, BLOCK_SIZE=1024)

    return output


def add_Scalar(self: torch.Tensor, other: float, alpha: float = 1) -> torch.Tensor:
    """Implements aten::add.Scalar"""
    return add_Tensor(self, torch.full_like(self, other), alpha)


def add_out(self: torch.Tensor, other: torch.Tensor, alpha: float = 1, *, out: torch.Tensor) -> torch.Tensor:
    """Implements aten::add.out"""
    result = add_Tensor(self, other, alpha)
    out.copy_(result)
    return out
```

## Output Requirements

**Important**: Output the complete Python code directly in your reply:

1. Code must be wrapped in a ```python ... ``` code block
2. Code must be runnable as-is, without modification
3. Do not include test code or benchmark code
4. Do not add extra explanations, output only the code block
5. **Do not write code to a file**, output it directly in your reply

Example output format:
```python
import torch
import triton
import triton.language as tl

# Your implementation...
```
