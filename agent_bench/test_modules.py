"""Shared test module resolution for verify.py and verify_single.py."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default test module paths - can be overridden via config
DEFAULT_TEST_MODULES = {
    "v2": ["src/flagbench/accuracy/test_v2_ops.py"],
    "v2_1": ["src/flagbench/accuracy/test_v2_1_ops_with_benchmark.py"],
    "cupy": ["src/flagbench/accuracy/cublas/test_cublas_ops.py"],
    "KernelGenBench": [
        "src/flagbench/accuracy/test_v2_1_ops_with_benchmark.py",
        "src/flagbench/accuracy/vllm13/",
        "src/flagbench/accuracy/cublas/",
    ],
    "KernelGenBench-aten": [
        "src/flagbench/accuracy/test_v2_1_ops_with_benchmark.py",
    ],
    "KernelGenBench-vllm": [
        "src/flagbench/accuracy/vllm13/",
    ],
    "KernelGenBench-cublas": [
        "src/flagbench/accuracy/cublas/",
    ],
    "KernelGenBench-nocublas": [
        "src/flagbench/accuracy/test_v2_1_ops_with_benchmark.py",
        "src/flagbench/accuracy/vllm13/",
    ],
}


def get_test_modules(dataset: str, config: dict = None) -> list[str]:
    """Get test module path(s) for dataset.

    Args:
        dataset: Dataset name (v2, v2_1, cupy, KernelGenBench, etc.)
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
