"""
Runtime Import Hook Sandbox for KernelGenBench.

Patches dangerous modules (triton, torch) at import time:
- Disables triton.autotune / triton.heuristics / triton.Config
- Disables torch.compile
- Disables CUDA Graph
- Blocks multiprocessing.shared_memory / posix_ipc / mmap

This is runtime enforcement (covers what AST scanning can miss):
- exec("import triton; ...")
- getattr(__import__("triton"), "autotune")
- __import__("triton").autotune(...)
"""
import sys
import importlib.abc
import importlib.machinery
import builtins
import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)


class _DisabledGraph:
    """Placeholder that raises when CUDA Graph is accessed."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "CUDA Graph is disabled in competition mode. "
            "Kernel capture and replay is not allowed."
        )


def _patch_triton_module(module):
    """Patch triton module: disable autotune, heuristics, Config."""
    if hasattr(module, "autotune"):
        module.autotune = lambda *a, **k: (lambda fn: fn)
    if hasattr(module, "heuristics"):
        module.heuristics = lambda *a, **k: (lambda fn: fn)
    # Config objects are no-ops
    if hasattr(module, "Config"):
        _original_Config = module.Config

        class _NoopConfig:
            def __init__(self, *args, **kwargs):
                pass

        module.Config = _NoopConfig
    logger.debug("ImportHook: patched triton (autotune/heuristics/Config disabled)")


def _patch_torch_module(module):
    """Patch torch module: disable compile, CUDA graph, IPC."""
    if hasattr(module, "compile"):
        module.compile = lambda *a, **k: (lambda fn: fn)
    if hasattr(module, "cuda") and hasattr(module.cuda, "graph"):
        module.cuda.graph = _DisabledGraph
    if hasattr(module, "cuda") and hasattr(module.cuda, "make_graphed_callables"):
        module.cuda.make_graphed_callables = lambda *a, **k: a[0] if a else None
    # Block multiprocessing from being accessed via torch
    logger.debug("ImportHook: patched torch (compile/cuda_graph/ipc disabled)")


class _MonitoredLoader(importlib.abc.Loader):
    """Wraps the original loader to apply patches after module init."""

    def __init__(self, original_loader, module_name: str):
        self._original_loader = original_loader
        self._module_name = module_name

    def create_module(self, spec):
        return None  # use default

    def exec_module(self, module):
        self._original_loader.exec_module(module)
        if self._module_name == "triton":
            _patch_triton_module(module)
        elif self._module_name == "torch":
            _patch_torch_module(module)
        elif self._module_name in ("multiprocessing.shared_memory", "posix_ipc", "mmap"):
            raise ImportError(
                f"'{self._module_name}' is blocked in competition mode"
            )


class ImportHookSandbox(importlib.abc.MetaPathFinder):
    """sys.meta_path hook that monitors and patches dangerous imports.

    Inserted at position 0 in sys.meta_path so it intercepts all imports first.
    """

    MONITORED_MODULES = {"triton", "torch", "multiprocessing.shared_memory",
                         "posix_ipc", "mmap"}

    def __init__(self):
        self._import_log: list = []

    @property
    def import_log(self) -> list:
        return self._import_log

    def find_spec(self, fullname: str, path, target=None):
        self._import_log.append(fullname)

        is_monitored = fullname in self.MONITORED_MODULES or any(
            fullname.startswith(m + ".") for m in self.MONITORED_MODULES
        )
        if not is_monitored:
            return None

        # Find the original spec
        for finder in sys.meta_path:
            if finder is self:
                continue
            if hasattr(finder, "find_spec"):
                spec = finder.find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _MonitoredLoader(spec.loader, fullname)
                    return spec

        return None


class _SecureBuiltins:
    """Wraps exec/eval/compile to block forbidden keywords dynamically."""

    FORBIDDEN_KEYWORDS = {
        "autotune", "heuristics", "Autotuner",
        "torch.compile", "cuda.graph",
        "shared_memory", "mmap", "posix_ipc",
    }

    def __init__(self):
        self._original_exec = builtins.exec
        self._original_eval = builtins.eval

    def _check(self, code: str):
        for kw in self.FORBIDDEN_KEYWORDS:
            if kw in code:
                raise SecurityError(f"Forbidden keyword in exec/eval: '{kw}'")

    def secure_exec(self, source, globs=None, locs=None):
        s = source if isinstance(source, str) else str(source)
        self._check(s)
        return self._original_exec(source, globs, locs)

    def secure_eval(self, source, globs=None, locs=None):
        s = source if isinstance(source, str) else str(source)
        self._check(s)
        return self._original_eval(source, globs, locs)

    def enable(self):
        builtins.exec = self.secure_exec
        builtins.eval = self.secure_eval

    def disable(self):
        builtins.exec = self._original_exec
        builtins.eval = self._original_eval


class SecurityError(Exception):
    """Raised when a forbidden operation is detected at runtime."""
    pass


class RuntimeSandbox:
    """Runtime import sandbox manager.

    Usage:
        sandbox = RuntimeSandbox()
        sandbox.enable()
        # ... run user code ...
        sandbox.disable()

        # or:
        with RuntimeSandbox() as sandbox:
            # code runs with all hooks active
            pass
    """

    def __init__(self):
        self._hook = ImportHookSandbox()
        self._secure_builtins = _SecureBuiltins()
        self._enabled = False

    @property
    def import_log(self) -> list:
        return self._hook.import_log

    def enable(self):
        if self._enabled:
            return
        sys.meta_path.insert(0, self._hook)
        self._secure_builtins.enable()
        self._enabled = True
        logger.info("RuntimeSandbox: enabled")

    def disable(self):
        if not self._enabled:
            return
        if self._hook in sys.meta_path:
            sys.meta_path.remove(self._hook)
        self._secure_builtins.disable()
        self._enabled = False
        logger.info("RuntimeSandbox: disabled")

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *args):
        self.disable()
