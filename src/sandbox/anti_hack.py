"""
Anti-hack detection module for kernelgenbench.

Three layers of defense:
1. Static AST scan: whitelist-based torch API detection + forbidden import detection
2. Dual-execution comparison: disable triton.jit and re-run, if results unchanged -> hack
3. GPU profiling fingerprint: compare launched kernel names against expected patterns
"""

import ast
import logging
from typing import Tuple, List, Callable, Any, Set
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ============================================================
# Layer 1a: Whitelist — allowed torch.* API calls
# ============================================================
# Any torch.X() call NOT in this whitelist is considered a hack.
# Only tensor creation, dtype/size helpers, and triton API are allowed.

ALLOWED_TORCH_API: Set[str] = {
    # Tensor creation
    "torch.empty", "torch.zeros", "torch.ones", "torch.randn", "torch.rand",
    "torch.randint", "torch.arange", "torch.linspace", "torch.logspace",
    "torch.tensor", "torch.as_tensor", "torch.from_numpy",
    "torch.empty_like", "torch.zeros_like", "torch.ones_like", "torch.randn_like",
    "torch.full", "torch.full_like",
    # Dtype / type
    "torch.float16", "torch.bfloat16", "torch.float32", "torch.float64",
    "torch.int8", "torch.uint8", "torch.int16", "torch.int32", "torch.int64",
    "torch.bool", "torch.long", "torch.half", "torch.double",
    "torch.Size",
    # Constants
    "torch.pi", "torch.inf", "torch.nan", "torch.e",
    # Device / CUDA setup
    "torch.cuda.current_device", "torch.cuda.synchronize",
    "torch.cuda.device_count", "torch.cuda.get_device_name",
    "torch.cuda.get_device_capability", "torch.cuda.is_available",
    "torch.cuda.device",
    # Global flags
    "torch.no_grad", "torch.enable_grad", "torch.set_grad_enabled",
    "torch.manual_seed", "torch.set_default_device", "torch.set_default_dtype",
    "torch._C._cuda_getDeviceCount",  # used in triton internals
    # Triton API
    "triton.next_power_of_2", "triton.cdiv", "triton.autotune",
    "triton.Config", "triton.heuristics",
    "triton.language", "tl.",  # triton language operations
}

# triton.language.* and tl.* are special: any call starting with these prefixes
# is allowed (they represent thousands of triton operations)
_TRITON_ALLOWED_PREFIXES = ("triton.language.", "tl.", "triton.")

# ============================================================
# Layer 1b: Hard blacklist — always-forbidden imports/modules
# ============================================================

# Per-backend hard blacklists (always forbidden regardless of whitelist)
BACKEND_BLACKLISTS = {
    "vllm": ["vllm", "torch.ops.vllm"],
    "vllm13": ["vllm", "torch.ops.vllm"],
    "vllm15": ["vllm", "torch.ops.vllm"],
    "cublas": ["cupy", "cublas", "ctypes"],
    "torch": ["torch.ops.aten"],
    # SGLang backends
    "sglang": [],  # SGLang backend: harder to define specific blacklist beyond whitelist
}


class HackDetector(ast.NodeVisitor):
    """AST visitor that detects hack patterns in generated code.

    Three checks:
    1. Hard blacklist: forbidden imports/attribute access (ctypes, vllm, etc.)
    2. Torch API whitelist: any torch.X() call NOT in ALLOWED_TORCH_API -> hack
    3. getattr(torch, "xxx") / import alias detection: catch obfuscation attempts
    """

    def __init__(self, blacklist: List[str] = None):
        self.violations: List[str] = []
        self.blacklist = blacklist or []
        # Track import aliases: {"tr": "torch", "ts": "torch.sum", ...}
        self._aliases: dict = {}

    # ---- Hard blacklist + alias tracking: imports ----
    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            if self._is_blacklisted(alias.name):
                self.violations.append(
                    f"Forbidden import: 'import {alias.name}' (line {node.lineno})"
                )
            # Track alias: import torch as tr -> "tr" -> "torch"
            if alias.asname:
                self._aliases[alias.asname] = alias.name
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module and self._is_blacklisted(node.module):
            names = ", ".join(a.name for a in node.names)
            self.violations.append(
                f"Forbidden import: 'from {node.module} import {names}' (line {node.lineno})"
            )
        # Track aliases: from torch import sum as ts -> "ts" -> "torch.sum"
        if node.module:
            for alias in node.names:
                key = alias.asname or alias.name
                self._aliases[key] = f"{node.module}.{alias.name}"
        self.generic_visit(node)

    # ---- Torch API whitelist + getattr detection: call detection ----
    def visit_Call(self, node: ast.Call):
        # Get the full attribute chain of the call, e.g. "torch.sum"
        call_chain = self._get_attr_chain(node.func)
        call_chain = self._resolve_alias(call_chain) if call_chain else call_chain

        if call_chain:
            # Check hard blacklist first
            if self._is_blacklisted(call_chain):
                self.violations.append(
                    f"Forbidden call: '{call_chain}()' (line {node.lineno})"
                )
            # Check torch API whitelist
            elif self._is_torch_api(call_chain) and not self._is_allowed(call_chain):
                self.violations.append(
                    f"Forbidden torch API: '{call_chain}()' not in allowed whitelist (line {node.lineno})"
                )

        # Detect getattr(torch, "sum") — dynamic attribute access
        if (
            isinstance(node.func, ast.Name) and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            # Reconstruct: getattr(torch, "sum") -> "torch.sum"
            base = self._get_attr_chain(node.args[0])
            if base:
                full = f"{base}.{node.args[1].value}"
                full = self._resolve_alias(full)
                if self._is_torch_api(full) and not self._is_allowed(full):
                    self.violations.append(
                        f"Forbidden torch API via getattr: 'getattr({base}, \"{node.args[1].value}\")' (line {node.lineno})"
                    )

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

    # ---- Hard blacklist: attribute access ----
    def visit_Attribute(self, node: ast.Attribute):
        attr_chain = self._get_attr_chain(node)
        if attr_chain and self._is_blacklisted(attr_chain):
            self.violations.append(
                f"Forbidden attribute access: '{attr_chain}' (line {node.lineno})"
            )
        self.generic_visit(node)

    # ---- Helpers ----
    def _is_blacklisted(self, module_name: str) -> bool:
        for prefix in self.blacklist:
            if module_name == prefix or module_name.startswith(prefix + "."):
                return True
        return False

    def _is_torch_api(self, call_chain: str) -> bool:
        """Check if this is a torch.* function call (as opposed to method call)."""
        return call_chain.startswith("torch.")

    def _is_allowed(self, call_chain: str) -> bool:
        """Check if a call chain is in the allowed whitelist."""
        # Exact match
        if call_chain in ALLOWED_TORCH_API:
            return True
        # Prefix match for triton/tl
        for prefix in _TRITON_ALLOWED_PREFIXES:
            if call_chain.startswith(prefix):
                return True
        return False

    def _resolve_alias(self, call_chain: str) -> str:
        """Resolve import aliases: 'tr.sum' -> 'torch.sum'."""
        if not call_chain:
            return call_chain
        parts = call_chain.split(".")
        first = parts[0]
        if first in self._aliases:
            parts[0] = self._aliases[first]
            return ".".join(parts)
        return call_chain

    def _is_importlib_call(self, node: ast.Call) -> bool:
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "import_module":
            if isinstance(func.value, ast.Name) and func.value.id == "importlib":
                return True
        return False

    def _get_attr_chain(self, node: ast.AST) -> str:
        """Reconstruct 'torch.nn.functional.relu' from AST Attribute nodes."""
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.Call):
            # chained call like foo().bar — skip
            return None
        else:
            return None
        return ".".join(reversed(parts))


def check_code(code: str, backend: str = None) -> Tuple[bool, str]:
    """
    Check if generated code contains hack patterns.

    Two-pronged check:
    1. Torch API whitelist (always active): any torch.X() call not in ALLOWED_TORCH_API -> hack
    2. Backend-specific hard blacklist: forbidden imports/access (ctypes, vllm, etc.)

    Args:
        code: generated triton kernel source code
        backend: one of "vllm", "vllm13", "vllm15", "cublas", "torch", "sglang".
                 If provided, also checks backend-specific hard blacklist.

    Returns:
        (is_hack, reason): True if hack detected, with explanation.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, ""

    blacklist = BACKEND_BLACKLISTS.get(backend, []) if backend else []
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
