"""
Layer 2: Python Import Hook Sandbox (replaces AST scanning)
From: triton_competition_anti_cheat_guide.md - Section 4
"""
import sys
import importlib.abc
import importlib.machinery
import builtins
from typing import Optional, Set


class ForbiddenModuleLoader(importlib.abc.Loader):
    """禁止加载特定模块的Loader"""

    def __init__(self, forbidden_names: Set[str], original_loader):
        self.forbidden_names = forbidden_names
        self.original_loader = original_loader

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # 执行原始加载
        self.original_loader.exec_module(module)

        # 对特定模块应用patch
        module_name = module.__name__
        if module_name == 'triton':
            self._patch_triton(module)
        elif module_name == 'torch':
            self._patch_torch(module)

    def _patch_triton(self, triton_module):
        """Patch Triton模块"""
        # 禁用autotune
        if hasattr(triton_module, 'autotune'):
            triton_module.autotune = lambda *a, **k: (lambda fn: fn)
        if hasattr(triton_module, 'heuristics'):
            triton_module.heuristics = lambda *a, **k: (lambda fn: fn)
        if hasattr(triton_module, 'Config'):
            triton_module.Config = lambda *a, **k: None

    def _patch_torch(self, torch_module):
        """Patch PyTorch模块"""
        # 禁用torch.compile
        if hasattr(torch_module, 'compile'):
            torch_module.compile = lambda *a, **k: (lambda fn: fn)
        # 禁用CUDA Graph
        if hasattr(torch_module, 'cuda') and hasattr(torch_module.cuda, 'graph'):
            torch_module.cuda.graph = DisabledCUDAGraph()


class DisabledCUDAGraph:
    """禁用的CUDA Graph占位类"""

    def __init__(self):
        raise RuntimeError("CUDA Graph is disabled in competition mode")


class ImportHookSandbox(importlib.abc.MetaPathFinder):
    """Import Hook沙箱"""

    FORBIDDEN_PATTERNS = {
        'triton.autotune',
        'triton.heuristics',
        'triton.Autotuner',
        'torch.compile',
        'torch.cuda.graph',
        'multiprocessing.shared_memory',
        'posix_ipc',
        'mmap',
    }

    def __init__(self):
        self.patched_modules = set()
        self.import_log = []

    def find_spec(self, fullname, path, target=None):
        """拦截模块加载"""
        # 记录import
        self.import_log.append(fullname)

        # 检查是否为需要监控的模块
        monitored = {'triton', 'torch', 'multiprocessing', 'posix_ipc', 'mmap'}

        if fullname in monitored or any(fullname.startswith(m + '.') for m in monitored):
            # 找到原始loader
            for finder in sys.meta_path:
                if finder is self:
                    continue
                if hasattr(finder, 'find_spec'):
                    spec = finder.find_spec(fullname, path, target)
                    if spec is not None:
                        # 包装loader
                        spec.loader = ForbiddenModuleLoader(
                            self.FORBIDDEN_PATTERNS,
                            spec.loader
                        )
                        return spec

        return None


class RuntimeSandbox:
    """运行时沙箱管理器"""

    def __init__(self):
        self.hook = ImportHookSandbox()
        self.original_meta_path = None
        self._secure_builtins = SecureBuiltins()

    def enable(self):
        """启用沙箱"""
        self.original_meta_path = sys.meta_path.copy()
        sys.meta_path.insert(0, self.hook)
        self._secure_builtins.enable()

    def disable(self):
        """禁用沙箱"""
        if self in sys.meta_path:
            sys.meta_path.remove(self.hook)
        if self.original_meta_path:
            sys.meta_path = self.original_meta_path
        self._secure_builtins.disable()

    def get_import_log(self):
        """获取import日志"""
        return self.hook.import_log

    def __enter__(self):
        self.enable()
        return self

    def __exit__(self, *args):
        self.disable()


def enable_competition_sandbox():
    """一键启用竞赛沙箱"""
    sandbox = RuntimeSandbox()
    sandbox.enable()
    return sandbox


class SecureBuiltins:
    """安全的内置函数包装"""

    def __init__(self):
        self.original_exec = builtins.exec
        self.original_eval = builtins.eval
        self.original_compile = builtins.compile
        self.original_print = builtins.print

        # 禁止的关键字
        self.forbidden_keywords = {
            'autotune', 'heuristics', 'Autotuner',
            'torch.compile', 'cuda.graph',
            'shared_memory', 'mmap', 'posix_ipc',
        }

    def secure_exec(self, source, globals=None, locals=None):
        """安全的exec"""
        source_str = source if isinstance(source, str) else str(source)
        self._check_forbidden(source_str)
        return self.original_exec(source, globals, locals)

    def secure_eval(self, source, globals=None, locals=None):
        """安全的eval"""
        source_str = source if isinstance(source, str) else str(source)
        self._check_forbidden(source_str)
        return self.original_eval(source, globals, locals)

    def _check_forbidden(self, code: str):
        """检查禁止的关键字"""
        for keyword in self.forbidden_keywords:
            if keyword in code:
                raise SecurityError(f"Forbidden keyword detected: {keyword}")

    def enable(self):
        """启用安全包装"""
        builtins.exec = self.secure_exec
        builtins.eval = self.secure_eval
        # Disable print() to prevent input sniffing
        builtins.print = self._print_noop

    def disable(self):
        """禁用安全包装"""
        builtins.exec = self.original_exec
        builtins.eval = self.original_eval
        builtins.print = self.original_print

    @staticmethod
    def _print_noop(*args, **kwargs):
        """Print is disabled during evaluation."""
        pass


class SecurityError(Exception):
    """安全错误"""
    pass
