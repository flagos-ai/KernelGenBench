"""
Full Sandbox Orchestrator for KernelGenBench.

Combines all six defense layers into a single entry point:
1. File System Isolation  (cache_isolator.py)
2. Runtime Import Hook     (import_hook.py)
3. CUDA Layer Protection   (cuda_protector.py)
4. Bucketed Random Shape   (shape_generator.py)
5. Process Isolation       (process_isolator.py)
6. Statistical Timing      (timing_validator.py)

Plus the AST-level anti-hack check (anti_hack.py).
"""
import os
import logging
from typing import Callable, Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class FullSandbox:
    """Orchestrates all sandbox layers for competition-grade evaluation.

    Usage:
        sandbox = FullSandbox()

        # Run a single test with all layers active
        result = sandbox.evaluate(my_kernel_fn, input_a, input_b)
        if result["status"] == "success":
            print(f"Time: {result['stats']['median_ms']:.3f}ms")

        # Batch evaluation
        for shape in shapes:
            a, b = make_inputs(shape)
            result = sandbox.evaluate(my_kernel_fn, a, b)
            sandbox.report(result)
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        # Timing validator thresholds
        self._cv_threshold = cfg.get("cv_threshold", 0.15)
        self._iqr_threshold = cfg.get("iqr_threshold", 0.3)
        # Test config
        self._warmup_runs = cfg.get("warmup_runs", 1)
        self._timing_runs = cfg.get("timing_runs", 4)
        self._retest_ratio = cfg.get("retest_ratio", 0.2)
        # Code check
        self._backend = cfg.get("backend", "torch")

    def evaluate(self,
                 test_fn: Callable,
                 *test_args,
                 code: str = None,
                 retest: bool = False,
                 **test_kwargs) -> Dict[str, Any]:
        """Run a full evaluation with all sandbox layers.

        Args:
            test_fn: the function to benchmark
            *test_args: positional args passed to test_fn
            code: source code string (for AST anti-hack check)
            retest: if True, runs twice for consistency check
            **test_kwargs: keyword args passed to test_fn

        Returns:
            Full evaluation result dict.
        """
        # === Layer 0: AST anti-hack check ===
        if code:
            from sandbox.anti_hack import check_code
            is_hack, reason = check_code(code, backend=self._backend)
            if is_hack:
                return {"status": "hack_detected", "reason": reason}

        # === Layers 1-4: Run in isolated subprocess ===
        from sandbox.process_isolator import ProcessIsolatedEvaluator
        evaluator = ProcessIsolatedEvaluator()

        result = evaluator.run(
            test_fn, *test_args,
            warmup_runs=self._warmup_runs,
            timing_runs=self._timing_runs,
            **test_kwargs,
        )

        if result.get("status") != "success":
            return result

        times = result["times"]

        # === Layer 6: Statistical timing validation ===
        from sandbox.timing_validator import StatisticalTimingValidator
        validator = StatisticalTimingValidator(
            cv_threshold=self._cv_threshold,
            iqr_threshold=self._iqr_threshold,
        )
        validation = validator.validate(times)

        output = {
            "status": "success" if validation.is_valid else "warning",
            "times": times,
            "stats": {
                "median_ms": sorted(times)[len(times) // 2] * 1000,
                "min_ms": min(times) * 1000,
                "max_ms": max(times) * 1000,
                "cv": validation.cv,
                "iqr_ratio": validation.iqr_ratio,
                "convergence": validation.convergence_score,
            },
            "anomaly": None if validation.is_valid else {
                "type": validation.anomaly_type.value,
                "message": validation.message,
            },
        }

        # === Retest consistency ===
        if retest:
            retest_result = evaluator.run(
                test_fn, *test_args,
                warmup_runs=self._warmup_runs,
                timing_runs=self._timing_runs,
                **test_kwargs,
            )
            if retest_result.get("status") == "success":
                is_cons, msg = validator.check_consistency(
                    times, retest_result["times"]
                )
                output["retest"] = {"is_consistent": is_cons, "message": msg}
                if not is_cons:
                    output["status"] = "warning"

        return output

    def report(self, result: Dict[str, Any]) -> str:
        """Format a one-line report for logging/display."""
        if result["status"] == "hack_detected":
            return f"[HACK] {result['reason'][:80]}"
        if result["status"] == "error":
            return f"[ERR] {result.get('error', 'unknown')[:80]}"
        s = result["stats"]
        line = f"[OK] {s['median_ms']:.3f}ms | CV={s['cv']:.3f} IQR={s['iqr_ratio']:.3f}"
        if result.get("anomaly"):
            line += f" | WARN: {result['anomaly']['type']}"
        return line


def enable_sandbox():
    """Convenience: enable in-process sandbox layers (for non-isolated use).

    Returns a context manager that enables import hooks + CUDA protection.
    Does NOT apply process isolation.
    """
    from contextlib import contextmanager

    @contextmanager
    def _sandbox_ctx():
        from sandbox.cache_isolator import CacheIsolator
        from sandbox.import_hook import RuntimeSandbox
        from sandbox.cuda_protector import CUDALayerProtector

        cache = CacheIsolator()
        cache.isolate()
        os.environ["TRITON_DISABLE_AUTOTUNE"] = "1"
        os.environ["CUDA_CACHE_DISABLE"] = "1"

        hook = RuntimeSandbox()
        hook.enable()

        cuda = CUDALayerProtector()
        cuda.setup()

        try:
            yield
        finally:
            cuda.restore()
            hook.disable()
            cache.cleanup()

    return _sandbox_ctx()
