#!/usr/bin/env python3
"""Batch verify generated Triton kernels."""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Add project paths
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    if yaml is None:
        print("Error: 'pyyaml' required. Install: pip install pyyaml")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_test_module(dataset: str, config: dict) -> str:
    """Get test module path for dataset."""
    test_modules = config.get("test_modules", {})
    if dataset in test_modules:
        return str(PROJECT_ROOT / test_modules[dataset])

    # Defaults
    defaults = {
        "v2": "src/flagbench/accuracy/test_v2_ops.py",
        "v2_1": "src/flagbench/accuracy/test_v2_1_ops_with_benchmark.py",
        "cupy": "src/flagbench/accuracy/cublas/test_cublas_ops.py",
    }
    if dataset in defaults:
        return str(PROJECT_ROOT / defaults[dataset])

    raise ValueError(f"No test module configured for dataset: {dataset}")


def verify_kernels(
    run_dir: Path,
    dataset: str,
    config: dict,
    device_count: int = 8,
    operators: list[str] | None = None,
    timeout: int = 600,
) -> dict:
    """Verify generated kernels using Verifier.

    Args:
        run_dir: Run directory containing kernels/
        dataset: Dataset name
        config: Config dict
        device_count: Number of GPUs
        operators: Optional specific operators to verify
        timeout: Timeout per operator

    Returns:
        Results dict
    """
    from sandbox.verifier.verifier import Verifier, VerifyConfig, VerifyRequest, Source

    kernels_dir = run_dir / "kernels"
    if not kernels_dir.exists():
        raise ValueError(f"Kernels directory not found: {kernels_dir}")

    # Get test module
    test_module = get_test_module(dataset, config)
    if not Path(test_module).exists():
        raise ValueError(f"Test module not found: {test_module}")

    # Setup environment
    os.environ["DISPATCH_TORCH_LIB"] = "1"
    os.environ["FLAGBENCH_SKIP_BOTH_TEST"] = "1"

    # Create verifier
    verify_config = VerifyConfig(
        run_name="",
        test_type="triton",
        run_dir=str(run_dir / "verification"),
        store_type="local",
        strict_check=True,
        seed=42,
        sample_id=0,
        save_log=True,
        acc_timeout=timeout,
    )

    verifier = Verifier(verify_config)
    verifier.set_modules(modules=[test_module], mode="accuracy")

    # Collect kernels to verify
    kernel_files = list(kernels_dir.glob("*.py"))
    if operators:
        filter_ops = set(operators)
        kernel_files = [f for f in kernel_files if f.stem in filter_ops]

    logger.info(f"Found {len(kernel_files)} kernels to verify")

    # Prepare verification requests
    verify_requests = []
    op_names = []

    # Determine namespace based on dataset
    namespace = "cupy" if dataset == "cupy" else "aten"

    for kernel_file in sorted(kernel_files):
        op_name = kernel_file.stem
        full_name = f"{namespace}::{op_name}"

        with open(kernel_file) as f:
            kernel_code = f.read()

        verify_req = VerifyRequest(
            source=[Source(
                source=kernel_code,
                function_name=full_name,
                namespace=""
            )],
            test_func=None,
        )
        verify_requests.append(verify_req)
        op_names.append(op_name)

    if not verify_requests:
        logger.warning("No kernels to verify")
        return {
            "run_name": run_dir.name,
            "dataset": dataset,
            "summary": {"total": 0, "passed": 0, "failed": 0, "pass_rate": "0%"},
            "operators": {},
        }

    # Run verification
    logger.info(f"Verifying {len(verify_requests)} kernels with {device_count} GPUs...")
    summary, results = verifier.only_verify(
        name_source_map=verify_requests,
        device_count=device_count,
    )

    # Process results
    # NOTE: Verifier uses mp.Queue which returns results in completion order,
    # not input order. Use result.op_name to map back to the correct operator.
    operators_results = {}
    passed = 0
    failed = 0

    for result in results:
        # Extract operator name from result (e.g., "aten::add" -> "add")
        result_op = result.op_name
        if "::" in result_op:
            result_op = result_op.split("::")[-1]

        op_result = {
            "status": "passed" if result.success else "failed",
            "total_tests": result.info.get("total", 0) if result.info else 0,
            "passed_tests": result.info.get("success", 0) if result.info else 0,
            "failed_tests": result.info.get("failed", 0) if result.info else 0,
        }

        if not result.success and result.traceback:
            # Truncate traceback
            tb = result.traceback
            if len(tb) > 500:
                tb = tb[:500] + "..."
            op_result["error"] = tb

        operators_results[result_op] = op_result

        if result.success:
            passed += 1
        else:
            failed += 1

    total = len(results)
    pass_rate = f"{passed / total * 100:.1f}%" if total > 0 else "0%"

    return {
        "run_name": run_dir.name,
        "dataset": dataset,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": pass_rate,
        },
        "operators": operators_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Batch verify generated Triton kernels")
    parser.add_argument(
        "--run", "-r",
        type=str,
        required=True,
        help="Run directory name to verify"
    )
    parser.add_argument(
        "--op", "-o",
        type=str,
        default=None,
        help="Specific operator(s) to verify, comma-separated"
    )
    parser.add_argument(
        "--device-count", "-d",
        type=int,
        default=8,
        help="Number of GPUs for verification (default: 8)"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=int,
        default=600,
        help="Timeout per operator in seconds (default: 600)"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override dataset (auto-detected from run if not specified)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config_path = args.config or (SCRIPT_DIR / "config.yaml")
    config = load_config(config_path)

    # Find run directory
    runs_dir = SCRIPT_DIR / config.get("paths", {}).get("runs", "runs")
    run_dir = runs_dir / args.run

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)

    # Load run config to get dataset
    dataset = args.dataset  # User override takes priority
    run_config_path = run_dir / "config.yaml"
    if run_config_path.exists():
        run_config = load_config(run_config_path)
        if not dataset:
            dataset = run_config.get("dataset")
    else:
        run_config = config

    # Try to get dataset from progress.json
    if not dataset:
        progress_path = run_dir / "progress.json"
        if progress_path.exists():
            with open(progress_path) as f:
                progress_data = json.load(f)
                dataset = progress_data.get("dataset")

    # Infer dataset from run name if needed
    if not dataset:
        run_name = args.run
        for ds in ["v2_1", "v2", "cupy"]:
            if ds in run_name:
                dataset = ds
                break

    if not dataset:
        print("Error: Could not determine dataset. Specify in config or run name.")
        sys.exit(1)

    # Parse operators
    operators = args.op.split(",") if args.op else None

    # Run verification
    logger.info(f"Verifying run: {args.run}")
    logger.info(f"Dataset: {dataset}")

    try:
        results = verify_kernels(
            run_dir=run_dir,
            dataset=dataset,
            config=config,
            device_count=args.device_count,
            operators=operators,
            timeout=args.timeout,
        )

        # Save results
        results_path = run_dir / "results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Print summary
        s = results["summary"]
        print(f"\n{'='*50}")
        print(f"Verification Results: {args.run}")
        print(f"Dataset: {dataset}")
        print(f"Total: {s['total']}, Passed: {s['passed']}, Failed: {s['failed']}")
        print(f"Pass Rate: {s['pass_rate']}")
        print(f"Results saved to: {results_path}")
        print(f"{'='*50}")

        # Print failed operators
        if s["failed"] > 0:
            print("\nFailed operators:")
            for op, info in results["operators"].items():
                if info["status"] == "failed":
                    error = info.get("error", "Unknown error")[:100]
                    print(f"  - {op}: {error}")

    except Exception as e:
        logger.error(f"Verification failed: {e}")
        raise


if __name__ == "__main__":
    main()
