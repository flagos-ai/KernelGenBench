"""
Process Isolation Execution for KernelGenBench.

Each test runs in a fresh subprocess with:
- Full sandbox layers applied (cache, import hook, CUDA, etc.)
- Clean global state (no module-level cache leaks)
- Independent memory space
"""
import multiprocessing as mp
import json
import time
import random
from typing import Callable, Dict, Any, List, Optional


def _isolated_worker(test_fn: Callable, test_args: tuple,
                    test_kwargs: dict, config: dict) -> Dict[str, Any]:
    """Worker function that runs inside an isolated subprocess.

    Applies all sandbox layers automatically before executing the test.
    """
    import torch
    import os

    # === Layer 1: File system isolation ===
    from sandbox.cache_isolator import CacheIsolator
    cache_isolator = CacheIsolator()
    cache_isolator.isolate()

    # === Layer 2: Environment variables ===
    os.environ["TRITON_DISABLE_AUTOTUNE"] = "1"
    os.environ["TORCHINDUCTOR_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    os.environ["CUDA_CACHE_DISABLE"] = "1"

    # === Layer 3: Import hook sandbox ===
    from sandbox.import_hook import RuntimeSandbox
    sandbox = RuntimeSandbox()
    sandbox.enable()

    # === Layer 4: CUDA protection ===
    from sandbox.cuda_protector import CUDALayerProtector
    cuda_protector = CUDALayerProtector()
    cuda_protector.setup()

    try:
        # Set seeds
        seed = config.get("seed", 42)
        torch.manual_seed(seed)
        random.seed(seed)

        # Warmup
        warmup = config.get("warmup_runs", 1)
        for _ in range(warmup):
            test_fn(*test_args, **test_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Timed runs
        times = []
        runs = config.get("timing_runs", 4)
        for _ in range(runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            test_fn(*test_args, **test_kwargs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        return {"status": "success", "times": times}

    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }

    finally:
        sandbox.disable()
        cuda_protector.restore()
        cache_isolator.cleanup()


class ProcessIsolatedEvaluator:
    """Runs tests in isolated subprocesses using spawn.

    Each evaluation gets a completely fresh process with:
    - Clean HOME and cache directories
    - Active runtime import hooks
    - CUDA protections enabled
    - No leaked global/static state

    Usage:
        evaluator = ProcessIsolatedEvaluator()
        result = evaluator.run(test_fn, input_a, input_b,
                               warmup_runs=1, timing_runs=4)
    """

    def __init__(self):
        self._ctx = mp.get_context("spawn")

    def run(self,
            test_fn: Callable,
            *test_args,
            warmup_runs: int = 1,
            timing_runs: int = 4,
            seed: int = None,
            **test_kwargs) -> Dict[str, Any]:
        """Run a single test in an isolated subprocess.

        Args:
            test_fn: callable to benchmark
            *test_args: positional args passed to test_fn
            warmup_runs: warmup iterations
            timing_runs: timed iterations
            seed: random seed (auto-generated if None)
            **test_kwargs: keyword args passed to test_fn

        Returns:
            {"status": "success", "times": [...]} or {"status": "error", ...}
        """
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        config = {
            "seed": seed,
            "warmup_runs": warmup_runs,
            "timing_runs": timing_runs,
        }

        with self._ctx.Pool(1) as pool:
            result = pool.apply(_isolated_worker,
                                (test_fn, test_args, test_kwargs, config))
        return result

    def run_batch(self,
                  test_fn: Callable,
                  args_list: List[tuple],
                  **kw) -> List[Dict[str, Any]]:
        """Run multiple tests, each in its own isolated subprocess."""
        results = []
        for args in args_list:
            result = self.run(test_fn, *args, **kw)
            results.append(result)
        return results
