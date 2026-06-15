# Custom Operators

How to add custom operators to KernelGenBench.

## Overview

To benchmark your own operators:
1. Add test cases to `src/kernelgenbench/accuracy/`
2. Register them in the dataset
3. Create operator schema

## Step 1: Create Accuracy Test

Create a file in `src/kernelgenbench/accuracy/`:

```python
# src/kernelgenbench/accuracy/custom_ops.py

import torch
from .base import AccuracyTest

class CustomAddTest(AccuracyTest):
    """Test for custom_add operator."""

    op_name = "custom::add"

    def get_test_cases(self):
        """Generate test cases for different shapes and dtypes."""
        return [
            # (shape, dtype, kwargs)
            ((64, 64), torch.float32, {}),
            ((128, 128), torch.float16, {}),
            ((256, 256), torch.float32, {}),
        ]

    def baseline(self, x, y):
        """Reference implementation."""
        return x + y

    def validate(self, output, expected, rtol=1e-5, atol=1e-5):
        """Validate output against expected."""
        return torch.allclose(output, expected, rtol=rtol, atol=atol)
```

## Step 2: Register Operator

Add to `src/kernelgenbench/dataset/registry.py`:

```python
OPERATOR_REGISTRY = {
    # ... existing operators ...
    "custom::add": {
        "test_class": "CustomAddTest",
        "module": "custom_ops",
        "description": "Custom addition operator",
    },
}
```

## Step 3: Create Schema (Optional)

For LLM prompt generation, create a schema:

```python
# src/kernelgenbench/schemas/custom.py

CUSTOM_ADD_SCHEMA = {
    "name": "custom::add",
    "inputs": [
        {"name": "x", "type": "Tensor", "description": "First input tensor"},
        {"name": "y", "type": "Tensor", "description": "Second input tensor"},
    ],
    "outputs": [
        {"name": "out", "type": "Tensor", "description": "Output tensor"},
    ],
    "description": "Element-wise addition of two tensors",
}
```

## Test Your Operator

```bash
python scripts/generate_kernel_and_verify.py \
    --op-name custom::add \
    --single-test \
    --server-type openai
```

## Best Practices

1. **Comprehensive test cases** - Cover edge cases
2. **Multiple dtypes** - Test float16, float32, etc.
3. **Various shapes** - Test small and large tensors
4. **Clear descriptions** - Help LLMs understand the operator
