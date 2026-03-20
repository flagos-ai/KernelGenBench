"""
Anti-hack detection module for flagbench.

Three layers of defense:
1. Static AST scan: detect forbidden imports/calls before execution
2. Dual-execution comparison: disable triton.jit and re-run, if results unchanged -> hack
3. GPU profiling fingerprint: compare launched kernel names against expected patterns
"""

import ast
import logging
from typing import Tuple, List, Callable, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Default blacklisted module prefixes
BLACKLISTED_MODULES = [
    "vllm",
    "torch.ops.vllm",
]

# Per-backend blacklists
BACKEND_BLACKLISTS = {
    "vllm": [
        "vllm",
        "torch.ops.vllm",
    ],
    "vllm13": [
        "vllm",
        "torch.ops.vllm",
    ],
    "vllm15": [
        "vllm",
        "torch.ops.vllm",
    ],
    "cublas": [
        "cupy",
        "cublas",
        "ctypes",
    ],
    "torch": [
        # torch backend: harder to define, only block direct aten dispatch
        "torch.ops.aten",
    ],
}


class HackDetector(ast.NodeVisitor):
    """AST visitor that detects hack patterns in generated code."""

    def __init__(self, blacklist: List[str] = None):
        self.violations: List[str] = []
        self.blacklist = blacklist or BLACKLISTED_MODULES

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if self._is_blacklisted(alias.name):
                self.violations.append(
                    f"Forbidden import: 'import {alias.name}' (line {node.lineno})"
                )
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module and self._is_blacklisted(node.module):
            names = ", ".join(a.name for a in node.names)
            self.violations.append(
                f"Forbidden import: 'from {node.module} import {names}' (line {node.lineno})"
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Detect __import__("vllm...")
        if isinstance(node.func, ast.Name) and node.func.id == "__import__":
            if node.args and isinstance(node.args[0], ast.Constant):
                if self._is_blacklisted(str(node.args[0].value)):
                    self.violations.append(
                        f"Forbidden dynamic import: '__import__(\"{node.args[0].value}\")' (line {node.lineno})"
                    )

        # Detect importlib.import_module("vllm...")
        if self._is_importlib_call(node):
            if node.args and isinstance(node.args[0], ast.Constant):
                if self._is_blacklisted(str(node.args[0].value)):
                    self.violations.append(
                        f"Forbidden dynamic import: 'importlib.import_module(\"{node.args[0].value}\")' (line {node.lineno})"
                    )

        # Detect exec() / eval()
        if isinstance(node.func, ast.Name) and node.func.id in ("exec", "eval"):
            self.violations.append(
                f"Forbidden call: '{node.func.id}()' (line {node.lineno})"
            )

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Detect torch.ops.vllm.xxx access
        attr_chain = self._get_attr_chain(node)
        if attr_chain:
            for blacklisted in self.blacklist:
                if attr_chain.startswith(blacklisted):
                    self.violations.append(
                        f"Forbidden attribute access: '{attr_chain}' (line {node.lineno})"
                    )
                    break
        self.generic_visit(node)

    def _is_blacklisted(self, module_name: str) -> bool:
        for prefix in self.blacklist:
            if module_name == prefix or module_name.startswith(prefix + "."):
                return True
        return False

    def _is_importlib_call(self, node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "import_module":
            if isinstance(func.value, ast.Name) and func.value.id == "importlib":
                return True
        return False

    def _get_attr_chain(self, node: ast.AST) -> str:
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        return ".".join(reversed(parts))


def check_code(code: str, backend: str = None) -> Tuple[bool, str]:
    """
    Check if generated code contains hack patterns.

    Args:
        code: generated triton kernel source code
        backend: one of "vllm", "vllm13", "vllm15", "cublas", "torch".
                 If provided, uses backend-specific blacklist.

    Returns:
        (is_hack, reason): True if hack detected, with explanation.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, ""

    blacklist = BACKEND_BLACKLISTS.get(backend) if backend else None
    detector = HackDetector(blacklist=blacklist)
    detector.visit(tree)

    if detector.violations:
        reason = "Anti-hack violations detected:\n" + "\n".join(
            f"  - {v}" for v in detector.violations
        )
        logger.warning(reason)
        return True, reason

    return False, ""


# ============================================================
# Layer 2: Dual-Execution Comparison (disable triton.jit)
# ============================================================

@contextmanager
def disable_triton_jit():
    """
    Context manager that makes all existing triton JIT kernels no-ops.
    Patches JITFunction.run so kernel invocations do nothing.
    """
    try:
        from triton.runtime.jit import JITFunction
        original_run = JITFunction.run
    except ImportError:
        yield
        return

    def noop_run(self, *args, **kwargs):
        return None

    JITFunction.run = noop_run
    try:
        yield
    finally:
        JITFunction.run = original_run


def dual_execution_check(
    func: Callable,
    kwargs: dict,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> Tuple[bool, str]:
    """
    Run func twice: once normally, once with triton.jit disabled.
    If results are the same, the code doesn't depend on triton kernels -> hack.

    Args:
        func: the registered triton function to test
        kwargs: input parameters for the function
        rtol/atol: tolerance for "same result" comparison

    Returns:
        (is_hack, reason): True if hack detected.
    """
    import torch

    # Run 1: normal execution
    try:
        out_normal = func(**kwargs)
    except Exception:
        # If normal run fails, can't do comparison
        return False, ""

    # Run 2: with triton.jit disabled
    try:
        with disable_triton_jit():
            # Re-import the module to pick up the patched triton.jit
            # The kernel functions become plain python functions
            out_disabled = func(**kwargs)
    except Exception:
        # If disabled run crashes, triton kernel was actually needed -> not hack
        return False, ""

    # Compare results
    if out_normal is None and out_disabled is None:
        # Both None - check in-place outputs via kwargs
        return False, ""

    if _results_match(out_normal, out_disabled, rtol, atol):
        return True, (
            "Dual-execution hack detected: output is identical "
            "with triton.jit disabled, indicating no real triton kernel is used."
        )

    return False, ""


def _results_match(a: Any, b: Any, rtol: float, atol: float) -> bool:
    """Check if two results are effectively identical."""
    import torch

    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        return torch.allclose(a.float(), b.float(), rtol=rtol, atol=atol)

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(_results_match(x, y, rtol, atol) for x, y in zip(a, b))

    return a == b


# ============================================================
# Layer 3: GPU Profiling Fingerprint
# ============================================================

def profile_kernel_names(func: Callable, kwargs: dict) -> Tuple[List[str], List[str]]:
    """
    Profile GPU kernel names launched during func execution.

    Returns:
        (all_cuda_kernels, cpu_launch_calls): CUDA kernel names and CPU-side launch calls.
    """
    import torch
    from torch.profiler import profile, ProfilerActivity

    # Warmup to avoid JIT compilation noise
    try:
        func(**kwargs)
        torch.cuda.synchronize()
    except Exception:
        pass

    cuda_kernels = []
    cpu_calls = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        func(**kwargs)
        torch.cuda.synchronize()

    for event in prof.key_averages():
        device_str = str(event.device_type).lower()
        if "cuda" in device_str:
            cuda_kernels.append(event.key)
        elif "cpu" in device_str:
            cpu_calls.append(event.key)

    return cuda_kernels, cpu_calls


# Known non-triton CUDA kernel patterns (native pytorch, cublas, etc.)
NON_TRITON_PATTERNS = [
    "at::native::",
    "cublas",
    "cudnn",
    "cufft",
    "void at::",
]


def gpu_profiling_check(
    func: Callable,
    kwargs: dict,
    expected_kernel_name: str = None,
) -> Tuple[bool, str]:
    """
    Profile GPU kernels during execution. Check if launched kernels
    are native CUDA ops (not triton) by comparing kernel name patterns.

    Args:
        func: function to profile
        kwargs: input parameters
        expected_kernel_name: if provided, check this name exists in CUDA kernels

    Returns:
        (is_hack, reason): True if hack detected.
    """
    try:
        cuda_kernels, cpu_calls = profile_kernel_names(func, kwargs)
    except Exception as e:
        logger.warning(f"GPU profiling failed: {e}")
        return False, ""

    if not cuda_kernels:
        return True, "GPU profiling hack detected: no CUDA kernels launched at all."

    # Check if ALL cuda kernels are known non-triton patterns
    all_non_triton = all(
        any(pat in name for pat in NON_TRITON_PATTERNS)
        for name in cuda_kernels
    )

    if all_non_triton:
        return True, (
            f"GPU profiling hack detected: all launched kernels are native CUDA ops, "
            f"no triton kernel found. Kernels: {cuda_kernels}"
        )

    return False, ""
