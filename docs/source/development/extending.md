# Extending the Framework

How to extend KernelGenBench with new platforms and methods.

## Adding a New Chip Backend

### Step 1: Add Device Detection

Edit `src/runtime/__init__.py`:

```python
def _detect_device_name() -> str:
    """Detect current hardware platform."""
    # Add detection logic for your platform
    if is_my_platform():
        return "my_platform"
    # ... existing detection logic ...

    return "unknown"
```

### Step 2: Create Platform Configuration

Add platform constraints to the `DEVICE_CONSTRAINTS` dict in `src/runtime/__init__.py`:

```python
class MyPlatformConfig:
    """Configuration for my_platform."""

    name = "my_platform"

    # Numerical tolerance
    rtol = 1e-4
    atol = 1e-4

    # Constraints
    max_threads_per_block = 1024

    # Import statements for generated code
    triton_import = """
import triton
import triton.language as tl
"""
```

### Step 3: Add Prompt Templates

Create `agent_bench/templates/my_platform/`:

````markdown
# Template for my_platform

{operator_schema}

Write a Triton kernel for this operator.
Use the following imports:

```python
{platform_imports}
```
````

### Step 4: Update Dependencies

Create `requirements/requirements_my_platform.txt`:

```
torch>=2.0
triton>=3.0
# Platform-specific packages
```

## Adding a New Agent Method

Follow these steps to integrate a new agent evaluation method into KernelGenBench.

### Step 1: Create Method Directory

```bash
mkdir -p agent_bench/methods/my_method/templates
```

### Step 2: Create Method Configuration

Create `agent_bench/methods/my_method/config.yaml`:

```yaml
name: my_method
description: My custom agent method
max_iterations: 10
timeout: 600
```

### Step 3: Create Instruction Template

Create `agent_bench/methods/my_method/templates/instructions.md`:

```markdown
# Task

Generate a Triton kernel for the following operator: {operator_name}

## Schema

{operator_schema}

## Requirements

1. Implement all functionality
2. Handle edge cases
3. Optimize for performance
```

### Step 4: Register Method

Edit `agent_bench/methods/__init__.py`:

```python
METHODS = {
    # ... existing methods ...
    "my_method": MyMethodRunner,
}
```

### Step 5: Create Test Script

Create `agent_bench/test_my_method.sh`:

```bash
#!/bin/bash
# Test script for my_method
python run.py --method my_method "$@"
```

## Adding New Evaluation Metrics

Add custom analysis to `scripts/analyze/analyze.py`:

```python
def compute_my_metric(results):
    """Compute custom metric."""
    # Implementation
    return value
```

## Testing Extensions

```bash
# Test new platform
python -c "from runtime import get_device_type; print(get_device_type())"

# Test new method
cd agent_bench && bash test_my_method.sh add --device-count 1
```
