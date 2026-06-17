# Custom Operators

How to add custom operators to KernelGenBench.

## Overview

To benchmark your own operators, follow the existing pattern (see vLLM/cuBLAS/SGLang as examples):

1. Add a thin baseline wrapper in `src/kernelgenbench/dataset/baseline/<namespace>/`
2. Register operator names in `src/kernelgenbench/dataset/kernel_list.py`
3. Create accuracy tests in `src/kernelgenbench/accuracy/<namespace>/`
4. (Optional) Add implementation info for prompt generation

## Step 1: Create Baseline Wrapper

Create a thin Python wrapper that calls your reference implementation.
Follow the vLLM pattern for simple function calls:

```python
# src/kernelgenbench/dataset/baseline/myns/my_op.py

import torch

try:
    from my_library import my_op_impl
except ModuleNotFoundError:
    my_op_impl = None


def my_op(
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Wrapper for my_op implementation."""
    return my_op_impl(x, y)
```

Create `src/kernelgenbench/dataset/baseline/myns/__init__.py` to export all operators:

```python
from .my_op import my_op
```

## Step 2: Register in Dataset

Add to `src/kernelgenbench/dataset/kernel_list.py`:

```python
MYNS_OPERATOR_NAMES = ['my_op']

def _load_myns_operators():
    from .baseline import myns
    return {f'myns::{name}': getattr(myns, name) for name in MYNS_OPERATOR_NAMES}

def get_myns_operators():
    return _load_myns_operators()
```

## Step 3: Create Accuracy Tests

Create accuracy tests following the vLLM/SGLang test pattern:

```python
# src/kernelgenbench/accuracy/myns/test_my_op.py

import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
import torch
import triton


@label("my_op")
@parametrize("shape", [(64, 64), (128, 128), (512, 512)])
@parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_accuracy_my_op(shape, dtype):
    M, N = shape
    x = torch.randn(M, N, device='cuda', dtype=dtype)
    y = torch.randn(M, N, device='cuda', dtype=dtype)

    ref_out = kernelgenbench.baseline.my_op(x, y)
    act_out = kernelgenbench.triton.my_op(x.clone(), y.clone())

    assert_close(act_out, ref_out, dtype)
```

## Step 4: Add Prompt Template (Optional)

For LLM-based kernel generation, add an operator specification to the dataset's
implementation info (see `vllm_IMPL_INFO.json` and `cublas_IMPL_INFO.json` for examples).

## Test Your Operator

```bash
# Test with a single operator
python scripts/generate_kernel_and_verify.py \
    --op-name myns::my_op \
    --single-test \
    --server-type openai

# Verify accuracy
# (runs the accuracy test you created in Step 3)
```
