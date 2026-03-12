base rules
rule1：中文回答
rule2：收到指令先用自然语言回答，先解释思路，不要直接动手，等我确定没问题下达指令再改代码
rule3:没有要求的情况下不要写md，直接输出回答就行
rule4：实验使用环境source /share/project/zhaohuxing/anaconda3/bin/activate zpy_flagbench
rule5:如果遇到提pr，不要你来推，你把推的终端命令告诉我，我来推，我来手动在github页面创建pr
rule6:就在指定的分支完成任务，千万不要干扰其他分支，其他已经提交pr的，或者还在开发的分支，都不要影响到！保持每个分支的干净独立！

# AGENTS.md - FlagBench Development Guide

## Project Overview
FlagBench is a benchmark framework for Triton kernel generation and verification. It supports automated test case generation, accuracy validation, and performance testing. The project is structured with source code in `src/`, tests in `test/`, and scripts in `scripts/`.

## Build, Lint, and Test Commands

### Installation
```bash
pip install -r requirements.txt
pip install .
```

### Type Checking
```bash
# Run pyright
pyright src/

# Run mypy
mypy src/
```

### Testing
```bash
# Run all tests
pytest

# Run a single test file
pytest test/test_imports.py

# Run a specific test class
pytest test/test_imports.py::TestImports

# Run a specific test function
pytest test/test_imports.py::TestImports::test_flagbench_imports

# Run tests with coverage
pytest --cov=src --cov-report=html
pytest --cov=src --cov-report=term-missing

# Run tests with specific marker
pytest -m unit
pytest -m integration
pytest -m gpu

# Run tests with timeout (requires pytest-timeout)
pytest --timeout=300

# Run accuracy tests for specific operators
python test/test_accuracy_ut.py --name abs
python test/test_accuracy_ut.py --name abs,mul,div
python test/test_accuracy_ut.py --name all

# Run with device count and timeout
python test/test_accuracy_ut.py --name abs --device-count 8 --timeout 300
```

### Code Formatting (if black/ruff is available)
```bash
# Format code (install black or ruff first if needed)
black src/ test/
# or
ruff check --fix src/ test/
```

## Code Style Guidelines

### Imports
- Organize imports in three groups: standard library, third-party, local package
- Use absolute imports for package modules
- Sort imports alphabetically within groups
- Example:
```python
import os
from typing import Callable, List, Optional

import torch
from rich.console import Console

from flagbench.dataset.kernel_list import PYTORCH_OPERATORS
from sandbox.register import Register
```

### Formatting
- Use 4 spaces for indentation
- Maximum line length: 120 characters (recommended)
- Use blank lines to separate logical sections within functions
- Use trailing commas in multi-line calls/definitions

### Type Hints
- Use type hints for function parameters and return values
- Use `Optional[T]` instead of `T | None` for Python 3.8 compatibility
- Use `List[T]`, `Dict[K, V]` from typing module
- Enable strict type checking in pyright: `typeCheckingMode = "basic"`
- Configure mypy with `python_version = "3.8"` and `strictParameterNoneValue = true`

### Naming Conventions
- **Classes**: PascalCase (e.g., `BaseGenerator`, `VerifyConfig`)
- **Functions/variables**: snake_case (e.g., `generate_prompt`, `test_verifier_operator`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `REPO_TOP_DIR`)
- **Private methods/variables**: leading underscore (e.g., `_init_data`, `_private_var`)
- **Type variables**: PascalCase (e.g., `T`, `K`, `V`)

### Dataclasses
- Use `@dataclass` for configuration objects and data containers
- Define field types explicitly
- Provide default values where appropriate
- Example:
```python
@dataclass
class VerifyConfig:
    run_name: str
    test_type: str = "accuracy"
    run_dir: str = os.path.join(REPO_TOP_DIR, "runs")
    seed: int = 42
    acc_timeout: int = 300
    perf_timeout: int = 600
```

### Error Handling
- Use try-except blocks with specific exception types
- Use logging instead of print statements for errors
- Define custom exceptions in `sandbox/error/error.py`
- Example:
```python
import logging
logger = logging.getLogger(__name__)

try:
    result = operation()
except ValueError as e:
    logger.error(f"Invalid value: {e}")
    raise
```

### Logging
- Use `logging` module with custom rich handler
- Configure logging in module initialization
- Log level format: `%(message)s`
- Example configuration:
```python
import logging
from .utils import CustomRichHandler

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[CustomRichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
```

### Testing Patterns
- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`
- Use pytest markers: `@pytest.mark.unit`, `@pytest.mark.gpu`, `@pytest.mark.slow`
- Use parametrized tests for multiple test cases
- Set `DISPATCH_TORCH_LIB = "0"` and `FLAGBENCH_UPCAST = "0"` in test files

### Project Structure
```
src/
  flagbench/          # Core benchmark functionality
  generator/          # Code generation modules
  sandbox/            # Verification and testing
test/                 # Test files
scripts/              # Utility scripts
FlagGems/             # FlagGems submodule
```

### Key Environment Variables
- `DISPATCH_TORCH_LIB`: Control Torch library dispatch (default: "1")
- `FLAGBENCH_USE_DYNAMIC_IMPL_INFO`: Enable dynamic implementation info
- `FLAGBENCH_SKIP_BOTH_TEST`: Skip double testing
- `FLAGBENCH_UPCAST`: Control type upcasting (default: "0")

### Important Paths
- Source code: `src/`
- Tests: `test/`
- Test paths in pytest.ini: `pythonpath = src`
- Output directories: `output/`, `output_ut/`, `runs/`, `cache/`
