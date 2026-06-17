"""
File System Isolation for KernelGenBench sandbox.

Ensures each test run has a clean filesystem:
- Isolated HOME directory (tmpfs)
- Disabled triton/torch/cuda caches
- Cleanup after each test
"""
import os
import shutil
import tempfile
import logging

logger = logging.getLogger(__name__)

# Cache paths that must be isolated or disabled per test run.
_CACHE_PATHS = [
    "~/.triton",
    "~/.cache/triton",
    "~/.torch",
    "~/.cache/torch",
    "~/.nv",
    "~/.cache/nvidia",
    "~/.cupy",
]


class CacheIsolator:
    """Cache directory isolator for safe kernel evaluation.

    Usage:
        with CacheIsolator() as home:
            # HOME is now a fresh tmpdir, all caches isolated
            run_test()

        # after context exit: HOME restored, tmpdir cleaned up
    """

    def __init__(self):
        self._original_home = None
        self._isolated_home = None

    def isolate(self) -> str:
        """Create an isolated environment. Returns the isolated HOME path."""
        self._original_home = os.environ.get("HOME", "/root")
        self._isolated_home = tempfile.mkdtemp(prefix="kgenbench_isolated_")

        os.environ["HOME"] = self._isolated_home

        # Point all caches into the isolated home (or disable)
        triton_cache = os.path.join(self._isolated_home, ".triton")
        inductor_cache = os.path.join(self._isolated_home, ".torch")
        xdg_cache = os.path.join(self._isolated_home, ".cache")

        os.makedirs(triton_cache, exist_ok=True)
        os.makedirs(inductor_cache, exist_ok=True)
        os.makedirs(xdg_cache, exist_ok=True)

        os.environ["TRITON_CACHE_DIR"] = triton_cache
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache
        os.environ["XDG_CACHE_HOME"] = xdg_cache
        os.environ["CUDA_CACHE_DISABLE"] = "1"

        logger.debug(f"CacheIsolator: HOME isolated to {self._isolated_home}")
        return self._isolated_home

    def cleanup(self):
        """Clean up the isolated environment and restore original HOME."""
        if self._isolated_home and os.path.exists(self._isolated_home):
            shutil.rmtree(self._isolated_home, ignore_errors=True)
            logger.debug(f"CacheIsolator: cleaned up {self._isolated_home}")
        if self._original_home:
            os.environ["HOME"] = self._original_home

    def __enter__(self):
        return self.isolate()

    def __exit__(self, *args):
        self.cleanup()
