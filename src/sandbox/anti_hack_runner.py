"""Anti-hack runner for batch checking operators."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from copy import deepcopy

from sandbox.anti_hack import check_code as anti_hack_check
from sandbox.verifier.verifier import Verifier, VerifyConfig, VerifyRequest, Source

logger = logging.getLogger(__name__)


class AntiHackRunner:
    """Runs anti-hack checks on operators with three-layer detection."""

    def __init__(self, dataset: str, verify_config: VerifyConfig,
                 custom_test_modules: Optional[List[str]] = None):
        """Initialize anti-hack runner.

        Args:
            dataset: Dataset name (e.g., "KernelGenBench")
            verify_config: Base verify config for Layer 2/3 checks
            custom_test_modules: Test modules to load (avoids test case bloat)
        """
        self.dataset = dataset
        self.verify_config = verify_config
        self.custom_test_modules = custom_test_modules

    def get_backend(self, op_name: str) -> str:
        """Determine anti-hack backend from operator name.

        Args:
            op_name: Operator name (e.g., "aten::add", "vllm13::fused_add")

        Returns:
            Backend name for anti-hack check ("torch", "vllm13", "cublas")
        """
        if op_name.startswith("vllm13::") or op_name.startswith("vllm15::"):
            return "vllm13"
        elif op_name.startswith("cublas::"):
            return "cublas"
        elif op_name.startswith("aten::"):
            return "torch"
        elif op_name.startswith("sglang::"):
            return "sglang"
        return self.dataset

    def need_namespace_triton(self, op_name: str) -> bool:
        """Check if operator needs namespace='triton' for verification.

        Args:
            op_name: Operator name

        Returns:
            True if namespace='triton' is needed
        """
        if self.dataset == "KernelGenBench" and not op_name.startswith("aten::"):
            return True
        return False

    def check_operator(
        self,
        op_name: str,
        kernel_code: str,
        kernel_path: Optional[Path] = None,
        test_file_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """Run three-layer anti-hack check on a single operator.

        Args:
            op_name: Operator name
            kernel_code: Kernel source code
            kernel_path: Path to kernel file (for Layer 2/3)
            test_file_path: Path to test file (for Layer 2/3)

        Returns:
            Dict with keys: hacked (bool), layer (str), reason (str)
        """
        backend = self.get_backend(op_name)

        # Layer 1: Static AST scan
        is_hack, reason = anti_hack_check(kernel_code, backend=backend)
        if is_hack:
            logger.warning(f"[Anti-hack L1] {op_name}: {reason}")
            return {"hacked": True, "layer": "L1", "reason": reason}

        # Layer 2 & 3: Dynamic execution check
        if kernel_path is None:
            # No kernel path provided, skip Layer 2/3
            return {"hacked": False, "layer": "", "reason": ""}

        if not kernel_path.exists():
            logger.warning(f"[Anti-hack] {op_name}: kernel file not found")
            return {"hacked": False, "layer": "", "reason": ""}

        try:
            # Create verify request
            with open(kernel_path) as f:
                code = f.read()

            verify_req = VerifyRequest(
                source=[Source(
                    source=code,
                    function_name=op_name,
                    namespace="triton" if self.need_namespace_triton(op_name) else ""
                )],
                test_func=str(test_file_path) if test_file_path and test_file_path.exists() else None,
            )

            # Run verification with anti_hack enabled
            verifier = Verifier(VerifyConfig(
                run_name="anti_hack",
                anti_hack=True,
                acc_timeout=self.verify_config.acc_timeout,
                save_log=False,
                manage_device_visibility=False,
            ))

            if self.custom_test_modules:
                verifier.set_modules(modules=self.custom_test_modules)

            _, results = verifier.only_verify(
                name_source_map=[verify_req],
                device_count=1,
            )

            if not results:
                return {"hacked": False, "layer": "", "reason": ""}

            result = results[0]
            if not result.success:
                # Check if failure is due to anti-hack detection
                if result.traceback and "anti" in result.traceback.lower():
                    logger.warning(f"[Anti-hack L2/L3] {op_name}: {result.traceback[:200]}")
                    return {"hacked": True, "layer": "L2/L3", "reason": result.traceback[:500]}

            return {"hacked": False, "layer": "", "reason": ""}

        except Exception as e:
            logger.warning(f"[Anti-hack] {op_name} check failed: {e}")
            return {"hacked": False, "layer": "", "reason": f"Check error: {str(e)}"}

    def batch_check(
        self,
        operators: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Run anti-hack checks on multiple operators.

        Args:
            operators: Dict mapping op_name to dict with keys:
                - kernel_code: str (required)
                - kernel_path: Path (optional, for Layer 2/3)
                - test_file_path: Path (optional, for Layer 2/3)

        Returns:
            Dict mapping op_name to hack result dict
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Anti-Hack Batch Check: {len(operators)} operators")
        logger.info(f"{'='*60}")

        results = {}
        for op_name, op_info in operators.items():
            kernel_code = op_info.get("kernel_code", "")
            if not kernel_code:
                logger.warning(f"[Anti-hack] {op_name}: no kernel code provided")
                continue

            result = self.check_operator(
                op_name=op_name,
                kernel_code=kernel_code,
                kernel_path=op_info.get("kernel_path"),
                test_file_path=op_info.get("test_file_path"),
            )
            results[op_name] = result

        # Summary
        hacked_count = sum(1 for r in results.values() if r.get("hacked"))
        clean_count = len(results) - hacked_count
        logger.info(f"\n{'='*60}")
        logger.info(f"Anti-Hack Summary: {clean_count}/{len(results)} clean, {hacked_count} hacked")
        logger.info(f"{'='*60}")

        return results

    def save_report(
        self,
        hack_results: Dict[str, Dict[str, Any]],
        total_operators: int,
        passed_operators: set,
        output_path: Path,
    ) -> Dict[str, Any]:
        """Save anti-hack report to JSON.

        Args:
            hack_results: Results from batch_check()
            total_operators: Total number of operators in dataset
            passed_operators: Set of operator names that passed verification
            output_path: Path to save the report JSON

        Returns:
            The report data dict
        """
        hacked_operators = {op for op, r in hack_results.items() if r["hacked"]}
        clean_passed = passed_operators - hacked_operators

        report = {
            "total_operators": total_operators,
            "original_passed": len(passed_operators),
            "hacked_count": len(hacked_operators),
            "clean_passed": len(clean_passed),
            "clean_pass_rate": len(clean_passed) / total_operators if total_operators else 0,
            "hacked_operators": sorted(list(hacked_operators)),
            "clean_passed_operators": sorted(list(clean_passed)),
            "hack_details": hack_results,
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Anti-hack results saved to: {output_path}")

        return report
