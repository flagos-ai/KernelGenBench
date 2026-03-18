#!/usr/bin/env python3
"""Verify a single Triton kernel for correctness.

This script is a shared verification tool used by multiple methods:
- normal_cc: CC calls this during self-verification
- iterative_optimizer: worker.py calls this during optimization loop

Usage:
    # Basic usage - results saved to output_dir/verify_logs/
    python verify_single.py --code v1/kernel.py --op add --output-dir v1/

    # JSON output to stdout
    python verify_single.py --code v1/kernel.py --op add --output-json
"""

import argparse
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Resolve paths relative to this script
# agent_bench/tools/verify_single.py
SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_BENCH_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = AGENT_BENCH_DIR.parent

# Add project paths
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# Default test module paths - can be overridden via config
DEFAULT_TEST_MODULES = {
    "v2": "src/flagbench/accuracy/test_v2_ops.py",
    "v2_1": "src/flagbench/accuracy/test_v2_1_ops_with_benchmark.py",
    "cupy": "src/flagbench/accuracy/cublas/test_cublas_ops.py",
}


def get_test_module(dataset: str, config: dict = None) -> str:
    """Get test module path for dataset.

    Args:
        dataset: Dataset name (v2, v2_1, cupy, etc.)
        config: Optional config dict with custom test_modules mapping

    Returns:
        Absolute path to test module

    Raises:
        ValueError: If no test module configured for dataset
    """
    # Check config first
    if config:
        test_modules = config.get("test_modules", {})
        if dataset in test_modules:
            return str(PROJECT_ROOT / test_modules[dataset])

    # Fall back to defaults
    if dataset in DEFAULT_TEST_MODULES:
        return str(PROJECT_ROOT / DEFAULT_TEST_MODULES[dataset])

    raise ValueError(f"No test module configured for dataset: {dataset}")


def _extract_avg_speedup(speedup_list):
    """Extract average speedup from speedup results list.

    Args:
        speedup_list: List of speedup measurement dicts from Verifier

    Returns:
        Average speedup value, or None if not available
    """
    if not speedup_list:
        return None
    # Find entry with params='avg' (Verifier adds this at the end)
    for item in reversed(speedup_list):
        if isinstance(item, dict) and item.get('params') == 'avg':
            return item.get('speedup')
    # Fallback: calculate average from all entries
    speedups = [item.get('speedup') for item in speedup_list
                if isinstance(item, dict) and isinstance(item.get('speedup'), (int, float))]
    return sum(speedups) / len(speedups) if speedups else None


def verify_single_kernel(
    code_path: Path,
    operator: str,
    dataset: str,
    timeout: int = 600,
    config: dict = None,
    output_dir: Path = None,
) -> dict:
    """Verify a single kernel for correctness.

    Args:
        code_path: Path to the kernel Python file
        operator: Operator name
        dataset: Dataset name (v2, v2_1, cupy)
        timeout: Timeout in seconds
        config: Optional config dict
        output_dir: Optional directory to save verification logs

    Returns:
        Dict with verification results:
        - passed: bool
        - total_tests: int
        - passed_tests: int
        - failed_tests: int
        - speedup: float or None
        - error: str or None
        - log_dir: str or None (path to saved logs if output_dir provided)
    """
    from sandbox.verifier.verifier import Verifier, VerifyConfig, VerifyRequest, Source

    if not code_path.exists():
        return {
            "passed": False,
            "error": f"File not found: {code_path}",
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
        }

    # Read kernel code
    with open(code_path) as f:
        kernel_code = f.read()

    # Get test module
    try:
        test_module = get_test_module(dataset, config)
    except ValueError as e:
        return {
            "passed": False,
            "error": str(e),
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
        }

    if not Path(test_module).exists():
        return {
            "passed": False,
            "error": f"Test module not found: {test_module}",
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
        }

    # Setup environment
    os.environ["DISPATCH_TORCH_LIB"] = "1"
    os.environ["FLAGBENCH_SKIP_BOTH_TEST"] = "1"

    # Determine where to save logs
    if output_dir:
        # Save logs to output_dir/
        run_dir = str(output_dir)
        save_log = True
        use_temp = False
        # Clean up old log directory to prevent Verifier from merging stale results
        old_log_dir = Path(run_dir) / "verify" / "log_0"
        if old_log_dir.exists():
            shutil.rmtree(old_log_dir)
    else:
        # Use temp directory (logs will be deleted)
        run_dir = tempfile.mkdtemp()
        save_log = False
        use_temp = True

    try:
        verify_config = VerifyConfig(
            run_name="verify",
            test_type="triton",
            run_dir=run_dir,
            store_type="local",
            strict_check=True,
            seed=42,
            sample_id=0,
            save_log=save_log,
            acc_timeout=timeout,
            manage_device_visibility=False,
        )

        verifier = Verifier(verify_config)
        verifier.set_modules(modules=[test_module], mode="accuracy")

        # Determine namespace based on dataset
        namespace = "cupy" if dataset == "cupy" else "aten"
        full_name = f"{namespace}::{operator}"

        # Prepare verification request
        verify_req = VerifyRequest(
            source=[Source(
                source=kernel_code,
                function_name=full_name,
                namespace=""
            )],
            test_func=None,
        )

        # Run verification (using single GPU - device_count=1)
        try:
            summary, results = verifier.only_verify(
                name_source_map=[verify_req],
                device_count=1,
            )

            result_dict = None
            if results and len(results) > 0:
                result = results[0]
                # Don't truncate traceback - CC needs full error info for debugging
                error_msg = result.traceback if result.traceback else None

                result_dict = {
                    "passed": result.success,
                    "total_tests": result.info.get("total", 0) if result.info else 0,
                    "passed_tests": result.info.get("success", 0) if result.info else 0,
                    "failed_tests": result.info.get("failed", 0) if result.info else 0,
                    "speedup": _extract_avg_speedup(result.speedup),
                    "error": error_msg,
                }
            else:
                result_dict = {
                    "passed": False,
                    "error": "No results returned from verifier",
                    "total_tests": 0,
                    "passed_tests": 0,
                    "failed_tests": 0,
                }

            # Add log directory path if saving logs
            if output_dir and save_log:
                # Verifier creates logs under run_dir/run_name/log_0/
                # run_name is "verify", so logs are at run_dir/verify/log_0/
                log_dir = Path(run_dir) / "verify" / "log_0"
                if log_dir.exists():
                    result_dict["log_dir"] = str(log_dir)

                    # Read speedup from result.json (Verifier saves it there, not in returned object)
                    result_json_path = log_dir / "result.json"
                    if result_json_path.exists() and result_dict.get("speedup") is None:
                        try:
                            with open(result_json_path) as f:
                                saved_results = json.load(f)
                            if saved_results and saved_results[0].get("speedup"):
                                result_dict["speedup"] = _extract_avg_speedup(saved_results[0]["speedup"])
                        except (json.JSONDecodeError, IOError, IndexError, KeyError):
                            pass  # Keep speedup as None if reading fails

                # Always save verify.json in output_dir for CC and worker to read
                verify_json_path = output_dir / "verify.json"
                with open(verify_json_path, "w") as f:
                    json.dump(result_dict, f, indent=2, ensure_ascii=False)

            return result_dict

        except Exception as e:
            # Don't truncate error - full info needed for debugging
            return {
                "passed": False,
                "error": str(e),
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
            }
    finally:
        # Clean up temp directory if used
        if use_temp and os.path.exists(run_dir):
            shutil.rmtree(run_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Verify a single Triton kernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic verification with logs saved
  python verify_single.py --code v1/kernel.py --op add --output-dir v1/

  # JSON output to stdout (for programmatic use)
  python verify_single.py --code v1/kernel.py --op add --output-json

  # Both: save logs and output JSON
  python verify_single.py --code v1/kernel.py --op add --output-dir v1/ --output-json
"""
    )
    parser.add_argument(
        "--code", "-c",
        type=Path,
        required=True,
        help="Path to kernel code file"
    )
    parser.add_argument(
        "--op", "-o",
        type=str,
        required=True,
        help="Operator name"
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="v2_1",
        help="Dataset name (default: v2_1)"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=600,
        help="Timeout in seconds (default: 600)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save verification logs (e.g., v1/). Creates verify.json and log_0/ subdirectory"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml for custom test module paths"
    )
    parser.add_argument(
        "--output-json",
        action="store_true",
        help="Output results as JSON to stdout"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.WARNING
    # When using --output-json, force all logging to stderr to avoid polluting JSON output
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr if args.output_json else None,
    )

    # Load config if provided
    config = None
    if args.config and args.config.exists():
        try:
            import yaml
            with open(args.config) as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    # Create output_dir if specified
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # When using --output-json, capture stdout to prevent pollution from Verifier
    if args.output_json:
        import io
        import contextlib
        captured_stdout = io.StringIO()
        with contextlib.redirect_stdout(captured_stdout):
            result = verify_single_kernel(
                code_path=args.code,
                operator=args.op,
                dataset=args.dataset,
                timeout=args.timeout,
                config=config,
                output_dir=args.output_dir,
            )
        # Log captured output to stderr if any
        captured = captured_stdout.getvalue()
        if captured:
            print(captured, file=sys.stderr)
        # JSON output for programmatic use
        print(json.dumps(result))
    else:
        result = verify_single_kernel(
            code_path=args.code,
            operator=args.op,
            dataset=args.dataset,
            timeout=args.timeout,
            config=config,
            output_dir=args.output_dir,
        )
        # Human-readable output
        if result["passed"]:
            print(f"PASSED: {args.op}")
            print(f"  Tests: {result['passed_tests']}/{result['total_tests']}")
            if result.get("speedup"):
                print(f"  Speedup: {result['speedup']:.2f}x")
            if result.get("log_dir"):
                print(f"  Logs: {result['log_dir']}")
        else:
            print(f"FAILED: {args.op}")
            if result.get("error"):
                # Truncate for display
                error_display = result['error'][:200]
                print(f"  Error: {error_display}")
            if result.get("log_dir"):
                print(f"  Logs: {result['log_dir']}")

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
