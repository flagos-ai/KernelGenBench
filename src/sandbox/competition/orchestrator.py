"""
Layer 7: Competition Anti-Cheat Orchestrator.

Ties together all 7 sandbox layers into a single pipeline.
This is the ONLY file users need to interact with.

Usage:
    from competition import run_check, CheckConfig

    result = run_check(
        kernel_path="path/to/kernel.py",
        generate_inputs=lambda: (torch.randn(128, 512, device='cuda'),),
    )
"""

import os
import sys
import time
import json
import random
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional, Tuple


# ============================================================
# Config
# ============================================================

@dataclass
class CheckConfig:
    """Anti-cheat check configuration."""
    kernel_path: str                                    # Path to kernel .py file
    generate_inputs: Callable[[], tuple]                # () -> (arg1, arg2, ...) or () -> {"kw1": v1, ...}
    num_tests: int = 10                                 # Number of test cases
    warmup_runs: int = 1                                # Warmup iterations
    timing_runs: int = 4                                # Measured iterations
    retest_ratio: float = 0.2                           # Fraction of cases to retest
    cv_threshold: float = 0.15                          # Coefficient of variation threshold
    iqr_threshold: float = 0.3                          # IQR anomaly threshold
    input_is_kwargs: bool = False                       # True if generate_inputs returns kwargs dict
    operator_name: str = "forward"                      # Function name to call in kernel module


# ============================================================
# Isolated worker (runs in subprocess with all sandbox layers)
# ============================================================

def _isolated_worker(config_dict: dict) -> dict:
    """
    Runs inside an isolated subprocess. All 7 sandbox layers are applied here.
    """
    import torch
    import numpy as np

    # ---- Layer 1: File system isolation ----
    from .cache_isolator import CacheIsolator
    cache_isolator = CacheIsolator()
    isolated_home = cache_isolator.isolate()

    # ---- Layer 2: Environment variables ----
    os.environ['HOME'] = isolated_home
    os.environ['TRITON_DISABLE_AUTOTUNE'] = '1'
    os.environ['TORCHINDUCTOR_DISABLE'] = '1'
    os.environ['TORCHDYNAMO_DISABLE'] = '1'
    os.environ['CUDA_CACHE_DISABLE'] = '1'
    os.environ['XDG_CACHE_HOME'] = f'{isolated_home}/.cache'

    # ---- Layer 3: Import hook sandbox ----
    from .import_hook import RuntimeSandbox
    sandbox = RuntimeSandbox()
    sandbox.enable()

    # ---- Layer 4: CUDA protection ----
    from .cuda_protector import CUDALayerProtector
    cuda_protector = CUDALayerProtector()
    cuda_protector.setup()

    try:
        # Load kernel
        kernel_path = config_dict['kernel_path']
        spec = __import__('importlib.util').util.spec_from_file_location(
            "competition_kernel", kernel_path
        )
        module = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(module)
        operator = getattr(module, config_dict.get('operator_name', 'forward'))

        # ---- Layer 5: Random shapes (via config_dict seeds) ----
        per_iteration_seeds = config_dict['per_iteration_seeds']
        warmup_seed = config_dict['warmup_seed']

        # Warmup
        torch.manual_seed(warmup_seed)
        random.seed(warmup_seed)
        np.random.seed(warmup_seed)
        warmup_inputs = config_dict['generate_inputs']()
        for _ in range(config_dict['warmup_runs']):
            if isinstance(warmup_inputs, dict):
                _ = operator(**{k: v.clone() if hasattr(v, 'clone') else v for k, v in warmup_inputs.items()})
            else:
                _ = operator(*[v.clone() if hasattr(v, 'clone') else v for v in warmup_inputs])
        torch.cuda.synchronize()

        # ---- Layer 6: Timing with per-iteration seeds ----
        times = []
        for i, iter_seed in enumerate(per_iteration_seeds):
            torch.manual_seed(iter_seed)
            random.seed(iter_seed)
            np.random.seed(iter_seed)

            inputs = config_dict['generate_inputs']()

            torch.cuda.synchronize()
            start = time.perf_counter()
            if isinstance(inputs, dict):
                _ = operator(**{k: v.clone() if hasattr(v, 'clone') else v for k, v in inputs.items()})
            else:
                _ = operator(*[v.clone() if hasattr(v, 'clone') else v for v in inputs])
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        return {
            'status': 'success',
            'times': times,
            'seed': config_dict['seed'],
        }

    except Exception as e:
        import traceback
        return {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc(),
            'seed': config_dict['seed'],
        }

    finally:
        sandbox.disable()
        cache_isolator.cleanup()


# ============================================================
# Layer 7: Orchestrator
# ============================================================

class CheckResult:
    """Result of an anti-cheat check."""
    def __init__(self, passed: bool, reason: str = "", details: dict = None):
        self.passed = passed
        self.reason = reason
        self.details = details or {}

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"CheckResult({status}, {self.reason[:60]})"

    def to_dict(self):
        return {
            "passed": self.passed,
            "reason": self.reason,
            "details": self.details,
        }


def run_check(
    kernel_path: str,
    generate_inputs: Callable[[], Any],
    *,
    operator_name: str = "forward",
    num_tests: int = 10,
    warmup_runs: int = 1,
    timing_runs: int = 4,
    retest_ratio: float = 0.2,
    cv_threshold: float = 0.15,
    iqr_threshold: float = 0.3,
    input_is_kwargs: bool = False,
    verbose: bool = True,
) -> CheckResult:
    """
    Run the full 7-layer anti-cheat check on a kernel.

    Args:
        kernel_path: Path to the .py file containing the kernel.
        generate_inputs: Callable that returns inputs for the kernel.
                         Can return tuple of positional args or dict of kwargs.
        operator_name: Name of the function to call in the kernel module.
        num_tests: Number of test cases to run.
        warmup_runs: Warmup iterations per test case.
        timing_runs: Measured iterations per test case.
        retest_ratio: Fraction of test cases to re-run for consistency.
        cv_threshold: Coefficient of variation threshold for timing anomalies.
        iqr_threshold: IQR threshold for outlier detection.
        input_is_kwargs: If True, generate_inputs returns a dict of kwargs.
        verbose: Print progress messages.

    Returns:
        CheckResult with .passed (bool) and .reason (str).
    """
    from .timing_validator import StatisticalTimingValidator

    if verbose:
        print("=" * 60)
        print("KernelGenBench Anti-Cheat Check — 7-Layer Sandbox")
        print("=" * 60)
        print(f"  Kernel: {kernel_path}")
        print(f"  Tests: {num_tests}, Timing runs: {timing_runs}")

    validator = StatisticalTimingValidator(
        cv_threshold=cv_threshold,
        iqr_threshold=iqr_threshold,
    )

    # Run tests in isolated subprocesses
    results = []
    for i in range(num_tests):
        seed = random.randint(0, 2**31 - 1)
        per_iteration_seeds = [
            seed + i * 10007 + random.randint(0, 99991)
            for i in range(timing_runs)
        ]

        config_dict = {
            'kernel_path': os.path.abspath(kernel_path),
            'generate_inputs': generate_inputs,
            'operator_name': operator_name,
            'seed': seed,
            'warmup_seed': seed - 1,
            'warmup_runs': warmup_runs,
            'timing_runs': timing_runs,
            'per_iteration_seeds': per_iteration_seeds,
            'input_is_kwargs': input_is_kwargs,
        }

        ctx = mp.get_context('spawn')
        with ctx.Pool(1) as pool:
            result = pool.apply(_isolated_worker, (config_dict,))

        results.append(result)

        if verbose:
            if result['status'] == 'success':
                validation = validator.validate(result['times'])
                status = "PASS" if validation.is_valid else "WARN"
                median = float(np.median(result['times'])) * 1000
                print(f"  [{i+1}/{num_tests}] {median:.3f}ms {status}")
                if not validation.is_valid:
                    print(f"         Warning: {validation.message}")
            else:
                print(f"  [{i+1}/{num_tests}] ERROR: {result.get('error', 'unknown')[:80]}")

    # ---- Retest validation ----
    retest_count = int(num_tests * retest_ratio)
    if retest_count > 0:
        if verbose:
            print(f"\n  Retesting {retest_count} cases for consistency...")
        retest_indices = random.sample(range(num_tests), min(retest_count, num_tests))
        for idx in retest_indices:
            if results[idx]['status'] != 'success':
                continue
            seed = random.randint(0, 2**31 - 1)
            per_iteration_seeds = [
                seed + i * 10007 + random.randint(0, 99991)
                for i in range(timing_runs)
            ]
            config_dict = {
                'kernel_path': os.path.abspath(kernel_path),
                'generate_inputs': generate_inputs,
                'operator_name': operator_name,
                'seed': seed,
                'warmup_seed': seed - 1,
                'warmup_runs': warmup_runs,
                'timing_runs': timing_runs,
                'per_iteration_seeds': per_iteration_seeds,
                'input_is_kwargs': input_is_kwargs,
            }
            ctx = mp.get_context('spawn')
            with ctx.Pool(1) as pool:
                retest = pool.apply(_isolated_worker, (config_dict,))

            if retest['status'] == 'success':
                is_consistent, msg = validator.retest_comparison(
                    results[idx]['times'], retest['times']
                )
                if not is_consistent:
                    if verbose:
                        print(f"  [WARN] Case {idx+1} retest inconsistent: {msg}")

    # ---- Summarize ----
    success_results = [r for r in results if r['status'] == 'success']
    errors = [r for r in results if r['status'] != 'success']

    if errors:
        return CheckResult(
            passed=False,
            reason=f"{len(errors)}/{num_tests} tests errored: {errors[0].get('error', 'unknown')[:100]}",
            details={"errors": errors, "total": num_tests},
        )

    # Validate all timing
    import numpy as np
    anomalies = []
    for r in success_results:
        validation = validator.validate(r['times'])
        if not validation.is_valid:
            anomalies.append(validation.message)

    if anomalies:
        return CheckResult(
            passed=False,
            reason=f"Timing anomalies detected: {anomalies[0]}",
            details={"anomalies": anomalies, "total": num_tests},
        )

    all_times = [float(np.median(r['times'])) * 1000 for r in success_results]
    avg_time = float(np.mean(all_times))
    std_time = float(np.std(all_times))
    cv = std_time / avg_time if avg_time > 0 else 0

    if cv > cv_threshold:
        return CheckResult(
            passed=False,
            reason=f"Timing CV too high: {cv:.3f} > {cv_threshold}",
            details={"cv": cv, "avg_ms": avg_time, "std_ms": std_time},
        )

    if verbose:
        print(f"\n  ALL CLEAR — avg {avg_time:.3f}ms, std {std_time:.3f}ms, CV {cv:.3f}")

    return CheckResult(
        passed=True,
        reason="All checks passed",
        details={
            "avg_ms": avg_time,
            "std_ms": std_time,
            "cv": cv,
            "total_tests": num_tests,
            "valid_tests": len(success_results),
        },
    )


# ============================================================
# CLI entry point
# ============================================================

def _cli_example():
    """
    Example CLI usage:
        python -m sandbox.competition run path/to/kernel.py

    For a real CLI, the caller should write their own input generator.
    This is just a stub showing the pattern.
    """
    import torch
    if len(sys.argv) < 3:
        print("Usage: python -m sandbox.competition run <kernel.py>")
        print()
        print("The kernel must define a 'forward' function and a 'TEST_INPUTS' list.")
        print("Example TEST_INPUTS:")
        print("  TEST_INPUTS = [")
        print("      lambda: (torch.randn(128, 512, device='cuda', dtype=torch.float16),),")
        print("      lambda: (torch.randn(256, 512, device='cuda', dtype=torch.float16),),")
        print("  ]")
        sys.exit(1)

    cmd = sys.argv[1]
    kernel_path = sys.argv[2]

    if cmd == 'run':
        spec = __import__('importlib.util').util.spec_from_file_location(
            "test_kernel", kernel_path
        )
        module = __import__('importlib.util').util.module_from_spec(spec)
        spec.loader.exec_module(module)

        test_inputs = getattr(module, 'TEST_INPUTS', None)
        if test_inputs is None:
            print("ERROR: kernel must define TEST_INPUTS list")
            sys.exit(1)

        for i, gen_fn in enumerate(test_inputs):
            print(f"\n--- Test input set {i+1} ---")
            result = run_check(
                kernel_path=kernel_path,
                generate_inputs=gen_fn,
                num_tests=10,
            )
            print(f"\nResult: {result}")
            if not result.passed:
                sys.exit(1)


if __name__ == '__main__':
    _cli_example()