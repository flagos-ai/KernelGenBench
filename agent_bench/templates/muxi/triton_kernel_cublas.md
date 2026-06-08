# Triton Kernel Implementation Task (cuBLAS, MetaX MUXI)

You need to implement a Triton kernel for a cuBLAS baseline function on MetaX (MUXI) GPUs.

## Task Information

- **Operator**: {{OPERATOR}}
- **Full Name**: {{FULL_NAME}}
- **GPU ID**: {{GPU_ID}}

## Environment

- **Hardware**: MetaX (MUXI) GPU
- **Software**: MACA SDK-based PyTorch
- All device commands must be prefixed with `MACA_VISIBLE_DEVICES={{GPU_ID}}`
- Python path: `{{PYTHON_PATH}}`

## MetaX MUXI GPU Requirements (MUST follow)

- Device type is `cuda` (standard PyTorch CUDA API). No special import needed beyond `import torch`
- MetaX GPUs use the MACA SDK, providing a CUDA-compatible interface but with different hardware architecture. Avoid relying on NVIDIA-specific hardware features (e.g. Tensor Core specific instructions)
- Some advanced Triton features may not be supported or may behave differently. Prefer basic Triton operations
- Use `allow_tf32=False` for `tl.dot` to ensure precision
- Some operators have limited bfloat16 support. When encountering precision issues, prefer float32 accumulation

## Baseline Function

The following is the cuBLAS baseline function (calling cuBLAS C API via ctypes) you need to reimplement using Triton:

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
- Use `allow_tf32=False` for matrix operations to maintain precision
- Handle BLAS parameters correctly (e.g. `incx`, `incy`, `lda`, `ldb`, `ldc`, `trans`, etc.)
- Note cuBLAS column-major storage convention

### 3. Test Environment

Your implementation will be tested as follows:
```python
# Baseline (cuBLAS C API wrapper)
from kernelgenbench.dataset.baseline.cublas.{{OPERATOR}} import {{OPERATOR}} as baseline_{{OPERATOR}}
ref_out = baseline_{{OPERATOR}}(...)

# Your Triton implementation
act_out = your_{{OPERATOR}}(...)

# Accuracy verification
assert_close(act_out, ref_out, dtype)
```

## Example

Below is the `saxpy` baseline function and its corresponding Triton implementation:

**cuBLAS baseline function:**
```python
def saxpy(n, alpha, x, incx, y, incy):
    '''SAXPY: y = alpha * x + y'''
    cublasSaxpy_v2(handle, n, alpha, x, incx, y, incy)
```

**Triton kernel implementation:**
```python
import torch
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
