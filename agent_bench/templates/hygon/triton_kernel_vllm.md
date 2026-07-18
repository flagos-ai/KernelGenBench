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

# Triton Kernel Implementation Task (vLLM, Hygon DCU)

You need to implement a Triton kernel for a vLLM baseline function.

## Task Information

- **Operator**: {{OPERATOR}}
- **Full Name**: {{FULL_NAME}}
- **GPU ID**: {{GPU_ID}}

## Environment

- **Hardware**: Hygon DCU (Deep Computing Unit)
- **Software**: ROCm/HIP-based PyTorch
- All device commands must be prefixed with `HIP_VISIBLE_DEVICES={{GPU_ID}}`
- Python path: `{{PYTHON_PATH}}`

## Hygon DCU Requirements (MUST follow)

- Device type is `cuda` (standard PyTorch CUDA API via ROCm/HIP). No special import needed beyond `import torch`
- Hygon DCU is based on the ROCm/HIP ecosystem, providing a CUDA-compatible interface. Avoid relying on NVIDIA-specific hardware features (e.g. Tensor Core instructions, CUDA-specific intrinsics)
- Some advanced Triton features may not be supported or may behave differently on the HIP backend. Prefer basic Triton operations
- Use `allow_tf32=False` for `tl.dot` to ensure precision (TF32 is an NVIDIA-specific feature)

## Baseline Function

The following is the vLLM baseline function you need to reimplement using Triton:

```python
{{BASELINE_CODE}}
```

### Function Signatures

{{OP_SIGNATURES}}

### Input/Output Arguments

{{INPUT_ARGS}}

## Implementation Requirements

### 1. Code Structure

Your implementation must include:
1. **Triton kernel function**: core computation logic decorated with `@triton.jit`
2. **Python wrapper function**: with the **exact same signature** as the baseline (function name, parameter names, parameter order must match)

### 2. Key Requirements

- Wrapper function signature must **exactly match** the baseline, otherwise tests will fail
- For float16/bfloat16 inputs, accumulate internally in float32
- Handle boundary conditions and edge cases correctly
- Respect in-place operations: if the baseline modifies input arguments (e.g. `out.copy_(...)`), your implementation must do the same

### 3. Test Environment

Your implementation will be tested as follows:
```python
# Baseline
from kernelgenbench.dataset.baseline.vllm13.{{OPERATOR}} import {{OPERATOR}} as baseline_fn
ref_out = baseline_fn(...)

# Your Triton implementation
act_out = your_{{OPERATOR}}(...)

# Accuracy verification
assert_close(act_out, ref_out, dtype)
```

## Example

Below is the `rms_norm` baseline function and its corresponding Triton implementation:

**vLLM baseline function:**
```python
def rms_norm_baseline(out, input, weight, epsilon):
    '''RMS normalization'''
    variance = input.pow(2).mean(-1, keepdim=True)
    input_normalized = input * torch.rsqrt(variance + epsilon)
    out.copy_(input_normalized * weight)
```

**Triton kernel implementation:**
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

## Output Requirements

**Important**: Output the complete Python code directly in your reply:

1. Code must be wrapped in a ```python ... ``` code block
2. Code must be runnable as-is, without modification
3. Include all necessary imports (torch, triton, etc.)
4. Include wrapper function with signature **exactly matching** the baseline
5. Do not include test code or benchmark code
6. Do not add extra explanations, output only the code block
7. **Do not write code to a file**, output it directly in your reply

Example output format:
```python
import torch
import triton
import triton.language as tl

# Your implementation...
```

{{REFERENCE_CODE}}
