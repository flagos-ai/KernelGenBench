# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared test module resolution for verify.py and verify_single.py."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default test module paths - can be overridden via config
DEFAULT_TEST_MODULES = {
    "KernelGenBench": [
        "src/kernelgenbench/accuracy/test_ops_with_benchmark.py",
        "src/kernelgenbench/accuracy/vllm13/",
        "src/kernelgenbench/accuracy/cublas/",
    ],
    "KernelGenBench-aten": [
        "src/kernelgenbench/accuracy/test_ops_with_benchmark.py",
    ],
    "KernelGenBench-vllm": [
        "src/kernelgenbench/accuracy/vllm13/",
    ],
    "KernelGenBench-cublas": [
        "src/kernelgenbench/accuracy/cublas/",
    ],
    "KernelGenBench-nocublas": [
        "src/kernelgenbench/accuracy/test_ops_with_benchmark.py",
        "src/kernelgenbench/accuracy/vllm13/",
    ],
    "MmShapeBench": [
        "src/kernelgenbench/accuracy/mm_shape_bench/",
    ],
}


def get_test_modules(dataset: str, config: dict = None) -> list[str]:
    """Get test module path(s) for dataset.

    Args:
        dataset: Dataset name (KernelGenBench, KernelGenBench-aten, etc.)
        config: Optional config dict with custom test_modules mapping

    Returns:
        List of absolute paths to test modules

    Raises:
        ValueError: If no test module configured for dataset
    """
    # Check config first
    if config:
        test_modules = config.get("test_modules", {})
        if dataset in test_modules:
            val = test_modules[dataset]
            if isinstance(val, list):
                return [str(PROJECT_ROOT / m) for m in val]
            return [str(PROJECT_ROOT / val)]

    # Fall back to defaults
    if dataset in DEFAULT_TEST_MODULES:
        return [str(PROJECT_ROOT / m) for m in DEFAULT_TEST_MODULES[dataset]]

    raise ValueError(f"No test module configured for dataset: {dataset}")


def get_test_module(dataset: str, config: dict = None) -> str:
    """Get test module path for dataset (backward compat, returns first module)."""
    modules = get_test_modules(dataset, config)
    return modules[0] if modules else None
