#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Test a single operator file - useful for debugging TLE optimization results

This script:
1. Takes a generated kernel file path as input
2. Reads the kernel code from the file
3. Uses kernelgenbench's test framework to verify the operator

Usage:
    # Basic usage
    python src/sandbox/server/test_single_operator.py path/to/aten_softmax.py

    # With custom test module
    python src/sandbox/server/test_single_operator.py path/to/aten_softmax.py \
        --test-module kernelgenbench.accuracy.test_ops_with_benchmark

    # With test set (auto-selects module)
    python src/sandbox/server/test_single_operator.py path/to/aten_softmax.py \
        --test-set KernelGenBench

    # Specify device
    CUDA_VISIBLE_DEVICES=7 python src/sandbox/server/test_single_operator.py \
        path/to/aten_softmax.py --device-count 1 --timeout 300
"""

import os
import sys
from pathlib import Path
import json
import argparse
import logging
from typing import Dict, List, Any

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # src/sandbox/server -> src/sandbox -> src -> project_root
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default test module for TLE optimization experiments
DEFAULT_TEST_MODULE = "kernelgenbench.accuracy.test_ops_with_benchmark"

# Default test set
DEFAULT_TEST_SET = "KernelGenBench"


# ============ Test Set Definitions ============

def get_test_sets() -> Dict[str, Dict[str, Any]]:
    """Get all test set configurations"""
    from kernelgenbench.dataset import get_kernelgenbench_operators
    return {
        "KernelGenBench": {
            "operators": get_kernelgenbench_operators(),
            "modules": [
                "kernelgenbench.accuracy.test_ops_with_benchmark",
                "kernelgenbench.accuracy.vllm13",
                "kernelgenbench.accuracy.cublas",
            ],
            "description": "KernelGenBench operators (210 operators)",
            "label_format": "{op}",
        },
    }


class SingleOperatorTester:
    """Test individual operators with generated kernels"""

    def __init__(
        self,
        kernel_file_path: str,
        output_dir: str = None,
        test_module: str = None,
        test_set: str = None
    ):
        """
        Initialize the tester

        Args:
            kernel_file_path: Path to the generated kernel .py file
            output_dir: Optional output directory for test results
            test_module: Test module to use (takes priority over test_set)
            test_set: Test set to use (KernelGenBench). Used if test_module is not specified.
        """
        self.kernel_file_path = Path(kernel_file_path)
        if not self.kernel_file_path.exists():
            raise FileNotFoundError(f"Kernel file not found: {kernel_file_path}")

        # Extract operator name from file path
        # Expected format: aten_operator_name.py -> aten::operator_name
        self.operator_name = self._extract_operator_name(self.kernel_file_path.stem)

        # Determine test module: --test-module takes priority, otherwise use --test-set
        if test_module:
            self.test_module = test_module
            self.test_set = None
            logger.info(f"Using explicit test module: {self.test_module}")
        else:
            # Use test_set to determine module
            self.test_set = test_set or DEFAULT_TEST_SET
            test_sets = get_test_sets()
            if self.test_set not in test_sets:
                raise ValueError(f"Unknown test set '{self.test_set}'. Available: {', '.join(test_sets.keys())}")
            self.test_module = test_sets[self.test_set]["modules"][0]
            logger.info(f"Using test set '{self.test_set}': {test_sets[self.test_set]['description']}")

        # Setup output directory
        if output_dir is None:
            output_dir = self.kernel_file_path.parent / "test_results"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized tester for operator: {self.operator_name}")
        logger.info(f"Kernel file: {self.kernel_file_path}")
        logger.info(f"Test module: {self.test_module}")
        logger.info(f"Output directory: {self.output_dir}")

    def _extract_operator_name(self, stem: str) -> str:
        """
        Extract operator name from file stem

        Examples:
            aten_softmax -> softmax
            aten__softmax -> _softmax
            softmax -> softmax
        """
        # Handle aten_ prefix
        if stem.startswith("aten_"):
            # aten_softmax -> softmax
            # aten__softmax -> _softmax (double underscore means the op name starts with _)
            op_name = stem[5:]  # Remove "aten_"
            return op_name
        elif "::" in stem:
            # aten::softmax -> softmax
            return stem.split("::")[-1]
        else:
            return stem

    def read_kernel_code(self) -> str:
        """Read the kernel code from the file"""
        logger.info(f"Reading kernel code from: {self.kernel_file_path}")
        with open(self.kernel_file_path, 'r') as f:
            code = f.read()

        logger.info(f"Kernel code length: {len(code)} characters")

        # Check if code contains tle.load
        if 'tle.load' in code:
            tle_count = code.count('tle.load')
            logger.info(f"Code contains {tle_count} tle.load calls")
        else:
            logger.info("Code does not contain tle.load")

        return code

    def create_verify_request(self, kernel_code: str) -> VerifyRequest:
        """
        Create a verification request for the operator

        Args:
            kernel_code: The kernel code to verify

        Returns:
            VerifyRequest object
        """
        return VerifyRequest(
            source=[Source(
                source=kernel_code,
                function_name=f"{self.operator_name}"
            )],
            test_func=None,
        )

    def run_test(self, device_count: int = 1, timeout: int = 300) -> dict:
        """
        Run the test for the operator

        Args:
            device_count: Number of devices to use for testing
            timeout: Timeout in seconds for each test

        Returns:
            Dictionary with test results
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing operator: {self.operator_name}")
        logger.info(f"{'='*60}")

        # Read kernel code
        kernel_code = self.read_kernel_code()

        # Setup environment
        os.environ["KERNELGENBENCH_USE_DYNAMIC_IMPL_INFO"] = "1"
        os.environ["KERNELGENBENCH_UPCAST"] = "1"
        os.environ["DISPATCH_TORCH_LIB"] = "1"
        os.environ["KERNELGENBENCH_SKIP_BOTH_TEST"] = "1"

        # Create verification config
        verify_config = VerifyConfig(
            run_name=f"tle_test_{self.operator_name}",
            test_type="triton",
            run_dir=str(self.output_dir),
            store_type="local",
            strict_check=True,
            seed=42,
            sample_id=0,
            save_log=True,
            acc_timeout=timeout,
            perf_timeout=timeout,
        )

        # Create verifier and set test module
        verifier = Verifier(verify_config)

        # Convert module name to path if needed
        if self.test_module.startswith("kernelgenbench."):
            # kernelgenbench.accuracy.test_ops_with_benchmark
            # -> src/kernelgenbench/accuracy/test_ops_with_benchmark.py
            module_path = self.test_module.replace(".", "/")
            module_path = f"src/{module_path}.py"
        else:
            module_path = self.test_module

        verifier.set_modules(
            modules=[module_path],
            mode="accuracy"
        )

        # Create verification request
        verify_request = self.create_verify_request(kernel_code)

        # Run verification
        logger.info(f"Running verification for {self.operator_name}...")
        try:
            _, results = verifier.only_verify(
                name_source_map=[verify_request],
                device_count=device_count,
            )

            # Process results
            if results and len(results) > 0:
                result = results[0]
                success = result.success
                logger.info(f"Test result: {'PASSED' if success else 'FAILED'}")

                # Load detailed test report
                # Result file path follows Verifier's convention: run_dir/run_name/log_{sample_id}/result.json
                result_file = self.output_dir / f"tle_test_{self.operator_name}" / "log_0" / "result.json"
                detailed_results = None
                if result_file.exists():
                    with open(result_file, 'r') as f:
                        detailed_results = json.load(f)

                return {
                    "operator": self.operator_name,
                    "kernel_file": str(self.kernel_file_path),
                    "success": success,
                    "result": result,
                    "detailed_results": detailed_results
                }
            else:
                logger.error("No results returned from verification")
                return {
                    "operator": self.operator_name,
                    "kernel_file": str(self.kernel_file_path),
                    "success": False,
                    "error": "No results returned"
                }

        except Exception as e:
            logger.error(f"Error during verification: {e}", exc_info=True)
            return {
                "operator": self.operator_name,
                "kernel_file": str(self.kernel_file_path),
                "success": False,
                "error": str(e)
            }

    def save_results(self, results: dict):
        """Save test results to JSON file"""
        results_file = self.output_dir / f"test_results_{self.operator_name}.json"

        # Convert non-serializable objects
        serializable_results = {
            "operator": results["operator"],
            "kernel_file": results["kernel_file"],
            "success": results["success"],
        }

        if "error" in results:
            serializable_results["error"] = results["error"]

        if "detailed_results" in results and results["detailed_results"]:
            serializable_results["detailed_results"] = results["detailed_results"]

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to: {results_file}")
        return results_file


def main():
    parser = argparse.ArgumentParser(
        description="Test a single operator with generated kernel (for TLE optimization debugging)"
    )

    parser.add_argument(
        "kernel_file",
        type=str,
        help="Path to the generated kernel .py file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for test results (default: kernel_file_dir/test_results)"
    )
    parser.add_argument(
        "--device-count",
        type=int,
        default=1,
        help="Number of devices to use for testing (default: 1)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for each test (default: 300)"
    )
    parser.add_argument(
        "--test-module",
        type=str,
        default=None,
        help=f"Test module to use (takes priority over --test-set)"
    )
    parser.add_argument(
        "--test-set",
        type=str,
        choices=["KernelGenBench"],
        default=DEFAULT_TEST_SET,
        help=f"Test set to use (default: {DEFAULT_TEST_SET}). Ignored if --test-module is specified."
    )

    args = parser.parse_args()

    # Create tester
    tester = SingleOperatorTester(
        kernel_file_path=args.kernel_file,
        output_dir=args.output_dir,
        test_module=args.test_module,
        test_set=args.test_set
    )

    # Run test
    results = tester.run_test(
        device_count=args.device_count,
        timeout=args.timeout
    )

    # Save results
    results_file = tester.save_results(results)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("Test Summary")
    logger.info(f"{'='*60}")
    logger.info(f"Operator: {results['operator']}")
    logger.info(f"Kernel file: {results['kernel_file']}")
    logger.info(f"Success: {results['success']}")
    if "error" in results:
        logger.info(f"Error: {results['error']}")
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"{'='*60}\n")

    # Exit with appropriate code
    sys.exit(0 if results['success'] else 1)


if __name__ == "__main__":
    main()
