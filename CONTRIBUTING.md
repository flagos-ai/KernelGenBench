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

# Contributing to KernelGenBench

We welcome contributions! Here's how to extend KernelGenBench in different ways.

## Adding New Operators (Test Cases)

To add a new operator to the benchmark dataset:

### 1. Create the accuracy test

Add a test file in `src/kernelgenbench/accuracy/`:

```python
# src/kernelgenbench/accuracy/test_my_op.py
import torch
from kernelgenbench import Param, parametrize, label

@parametrize(
    Param("shape", [(32, 64), (128, 256), (1024, 1024)]),
    Param("dtype", [torch.float16, torch.float32]),
)
@label("my_custom_op")
def test_my_custom_op(shape, dtype):
    """Test function for my_custom_op."""
    input = torch.randn(shape, dtype=dtype, device="cuda")
    # Run operator and compare with reference
    ...
```

### 2. Register in kernel_list

Add your operator to `src/kernelgenbench/dataset/kernel_list.py`:

```python
# In TORCH_OPERATOR_NAMES or create a new list
MY_OPERATORS = {
    "torch.ops.aten.my_op": torch.ops.aten.my_op,
}
```

### 3. Create a prompt template (optional)

If your operator needs special prompt context, add a prompt file in `agent_bench/prompts/KernelGenBench/`:

```
agent_bench/prompts/KernelGenBench/aten__my_op.md
```

### 4. Verify your test works

```bash
python scripts/generate_kernel_and_verify.py \
    --op-name aten::my_op \
    --single-test \
    --max-rounds 1
```

---

## Adding a New Chip Backend

To add support for a new hardware platform:

### 1. Update device detection

In `agent_bench/device_manager.py`, add your chip to `detect_device_type()`:

```python
if vendor == "my_chip" or os.environ.get("MY_CHIP_VISIBLE_DEVICES"):
    return "my_chip"
```

And add to `_VISIBLE_DEVICES_ENV`:

```python
_VISIBLE_DEVICES_ENV = {
    ...
    "my_chip": "MY_CHIP_VISIBLE_DEVICES",
}
```

### 2. Add device detection in DeviceManager

Add a `_detect_my_chip()` method to the `DeviceManager` class.

### 3. Add prompt templates

Create `agent_bench/templates/my_chip/triton_kernel_aten.md` and `triton_kernel_vllm.md` with chip-specific requirements (device type, import requirements, Triton limitations, etc.).

### 4. Add runtime constraints

In `src/runtime/__init__.py`, add your chip to `DEVICE_CONSTRAINTS`:

```python
DEVICE_CONSTRAINTS = {
    ...
    "my_chip": """
## Device-Specific Requirements
The operator runs on MyChip devices.
1. Device type is `cuda`/`my_chip`. ...
2. ...
3. ...
4. Triton notes: ...
""",
}
```

And add a detection helper `_is_my_chip()` and update `get_device_constraints()` and `get_device_type()`.

### 5. Test

```bash
GEMS_VENDOR=my_chip python scripts/generate_kernel_and_verify.py \
    --op-name aten::add --single-test --max-rounds 1
```

---

## Adding a New Agent (Coding Tool)

To add support for a new coding agent (e.g., Codex CLI, Trae, Cursor):

### 1. Create a new method directory

```
agent_bench/methods/naive_myagent/
├── __init__.py
├── method.py
└── templates/
    └── instructions.md
```

### 2. Implement the method

Your `method.py` must extend `BaseMethod`:

```python
from ..base import BaseMethod, MethodResult

class NaiveMyAgentMethod(BaseMethod):
    name = "naive_myagent"

    def launch(self, operator, prompt_path, workspace_dir, gpu_id, config, attempt=0):
        """Launch your agent process."""
        # Read prompt, set up environment, launch subprocess
        # Return a handle dict with at least {"proc": subprocess.Popen(...)}
        ...

    def finish(self, operator, handle, workspace_dir, config):
        """Extract generated code from agent output."""
        # Parse output, extract ```python``` code block
        # Return MethodResult(code=..., passed=None, speedup=None, metadata={})
        ...

    def get_process(self, handle):
        """Return the subprocess.Popen object."""
        return handle["proc"]
```

### 3. Register the method

In `agent_bench/methods/__init__.py`:

```python
from .naive_myagent import NaiveMyAgentMethod

_METHODS = {
    ...
    "naive_myagent": NaiveMyAgentMethod,
}
```

### 4. Test

```bash
cd agent_bench
bash test_ops.sh add -m naive_myagent --device-count 1
```

---

## Adding a New Agentic Method (Specialized Pipeline)

For more complex agent pipelines (like AutoKernel, AKO4ALL, cuda-optimized-skill):

### 1. Add your method to `sota_agents/`

```
sota_agents/my_method/
├── README.md
├── LICENSE
└── ...  (your method's code)
```

### 2. Create a run script

Create `agent_bench/run_my_method.py` that:
- Loads config and operator list
- Manages GPU allocation via `DeviceManager`
- Launches your pipeline per operator
- Saves kernels to `runs/<run_name>/kernels/`

### 3. Create a test script

Create `agent_bench/test_my_method.sh` following the pattern of `test_autokernel.sh`:
- Step 1: Generate prompts
- Step 2: Run your method
- Step 3: Verify results

### 4. Document

Add your method to the Methods table in `README.md`.

---

## Submitting Your Contribution

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-contribution`
3. Make your changes and ensure tests pass
4. Submit a Pull Request with a clear description

### PR Checklist

- [ ] Code has no sensitive information (API keys, internal URLs)
- [ ] All content is in English (code comments, templates, docs)
- [ ] `python -m py_compile` passes on modified files
- [ ] Tested with at least one operator end-to-end
- [ ] README updated if adding new user-facing features
