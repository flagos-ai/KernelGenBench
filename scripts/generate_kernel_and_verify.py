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

import os
os.environ["KERNELGENBENCH_USE_DYNAMIC_IMPL_INFO"] = "1"
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Set, List, Optional, Any
import sys
from datetime import datetime

from kernelgenbench.dataset import TorchOpsLoader, APIInfo
from kernelgenbench.dataset import IMPL_INFO
from sandbox.verifier import Verifier, VerifyConfig, VerifyRequest, Source
from utils import (
    today,
    _placeholder,
    create_triton_generate_args
)

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from generator import GENERATOR
from generator.sampler.generate_samples import (
    TritonKernelGenerateArgs,
    TestFuncGenerateArgs,
    BenchmarkFuncGenerateArgs,
    GenerationConfig,
)

def check_args_validity(args):
    assert args.test_type in ["triton"], "Invalid test type, must be triton."

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

mock_triton_code = "mock triton code"


from kernelgenbench.framework.kernelgenbench_adapter import KernelGenBenchAdapter
from generator.kernelgenbench_prompt_builder import KernelGenBenchPromptBuilder


class MmShapePromptBuilder:
    """MmShapeBench PromptBuilder — wraps KernelGenBenchPromptBuilder,
    temporarily swapping op_name to aten::mm for correct signatures."""

    def __init__(self, mode="basic"):
        self._inner = KernelGenBenchPromptBuilder(mode=mode)

    def build(self, gen_args):
        original = gen_args.triton_kernel_name
        gen_args.triton_kernel_name = "aten::mm"
        try:
            return self._inner.build(gen_args)
        finally:
            gen_args.triton_kernel_name = original


class PassAtKTester:
    def __init__(
        self,
        output_dir: Path,
        test_type: str,
        gen_config: GenerationConfig,
        verify_config: VerifyConfig,
        acc_test_func_path: str = "",
        bench_test_func_path: str = "",
        dataset: str = "KernelGenBench",
        custom_test_modules: Optional[List[str]] = None,
        device_count: int = 8,
        debug: bool = False,
        single_test: bool = False,
        op_name: Optional[str] = None,
        reflection: bool = False,
    ):
        self.output_dir = output_dir
        self.test_type = test_type
        self.gen_config = gen_config
        self.verify_config = verify_config
        self.acc_test_func_path = acc_test_func_path
        self.bench_test_func_path = bench_test_func_path
        self.dataset = dataset
        self.custom_test_modules = custom_test_modules
        self.device_count = device_count
        self.operator_loader = TorchOpsLoader()
        self.impl_info = IMPL_INFO
        self.create_generate_args = _placeholder
        self.create_verify_args = _placeholder
        self.adapter = None  # Framework adapter

        self.debug = debug
        self.single_test = single_test
        self.op_name = op_name
        self.reflection = reflection
        
        # Track results
        self.all_operators: Dict[str, Dict[str, APIInfo]] = {}
        self.passed_operators: Set[str] = set()
        self.results_by_round: List[Dict] = []
        self.generated_codes: Dict[str, Dict[int, str]] = {}  # {op_name: {round: code}}
        self.generation_summaries: List[Dict] = []  # Track generation summaries for each round
        self.verify_results: Dict[str, Dict[int, Dict]] = {}  # {op_name: {round: [test_results]}}

        self.setup()

    def setup(self):
        """Setup the os environment variables."""
        os.environ["KERNELGENBENCH_UPCAST"] = "0"
        if self.test_type == "triton":
            self.create_verify_args = self.create_triton_kernel_verify_args

            match self.dataset:
                case "KernelGenBench":
                    from kernelgenbench.dataset import get_kernelgenbench_operators
                    self.operator_loader = get_kernelgenbench_operators()
                case "KernelGenBench-aten":
                    from kernelgenbench.dataset import get_aten_operators
                    self.operator_loader = get_aten_operators()
                case "KernelGenBench-vllm":
                    from kernelgenbench.dataset import get_vllm_operators
                    self.operator_loader = get_vllm_operators()
                case "KernelGenBench-cublas":
                    from kernelgenbench.dataset import get_cublas_operators
                    self.operator_loader = get_cublas_operators()
                case "KernelGenBench-nocublas":
                    from kernelgenbench.dataset import get_kernelgenbench_nocublas_operators
                    self.operator_loader = get_kernelgenbench_nocublas_operators()
                case "MmShapeBench":
                    from kernelgenbench.dataset import get_mm_shape_operators
                    self.operator_loader = get_mm_shape_operators()
                case _:
                    raise ValueError(f"Unsupported dataset: {self.dataset}")

            # Create corresponding PromptBuilder and Adapter based on dataset
            self.prompt_builder = self._create_prompt_builder()
            self.adapter = self._create_adapter()

            self.create_generate_args = self._create_generate_args_wrapper()
        else:
            raise ValueError(f"Unsupported test type: {self.test_type}")

    def _create_prompt_builder(self):
        mode = "basic"
        if self.dataset == "MmShapeBench":
            prompt_builder = MmShapePromptBuilder(mode=mode)
            logger.info(f"Created MmShapePromptBuilder with mode: {mode}")
        else:
            prompt_builder = KernelGenBenchPromptBuilder(mode=mode)
            logger.info(f"Created KernelGenBenchPromptBuilder with mode: {mode}")
        return prompt_builder

    def _create_adapter(self):
        adapter = KernelGenBenchAdapter()
        logger.info("Created KernelGenBenchAdapter")
        return adapter

    def _create_generate_args_wrapper(self):
        """
        Create a generate_args wrapper function for KernelGenBench

        This wrapper adapts the calling convention in generate_round(),
        converting arguments to the format required by Adapter.create_generate_args

        Returns:
            Wrapper function
        """
        def wrapper(torch_op_name: str, torch_op_func_or_namespace: str, impl_info: Any):
            """
            Wrapper function, adapts the calling convention for KernelGenBench

            Args:
                torch_op_name: kernel name (e.g. "caxpy" or "softmax")
                torch_op_func_or_namespace: namespace (e.g. "vllm13", "cublas", "aten")
                impl_info: implementation info

            Returns:
                GenerateArgs instance
            """
            # For aten:: operators, use create_triton_generate_args directly
            if torch_op_func_or_namespace == "aten":
                return create_triton_generate_args(torch_op_name, torch_op_func_or_namespace, impl_info)

            # For vllm/cublas operators, use adapter
            # Construct full op_name (format: vllm13::rms_norm, cublas::sgemm, etc.)
            op_name = f"{torch_op_func_or_namespace}::{torch_op_name}"

            # Get function object from adapter
            func = self.adapter.get_operator_function(op_name)

            # Call adapter's create_generate_args method
            return self.adapter.create_generate_args(op_name, func, impl_info)

        return wrapper

    def initialize_operators(self, namespace: str = "all") -> None:
        """Initialize operator list, returns flat structure {op_name: value}"""
        if not isinstance(self.operator_loader, dict):
            # Dynamic loading - TorchOpsLoader (already returns flat structure)
            if namespace.lower() == "all":
                self.all_operators = self.operator_loader.load_all()
            else:
                self.all_operators = self.operator_loader.load_namespace(namespace)
        else:
            # Static loading - predefined dict (already flat)
            self.all_operators = self.operator_loader

        # Single operator mode
        if self.op_name:
            if self.op_name in self.all_operators:
                self.all_operators = {self.op_name: self.all_operators[self.op_name]}
                logger.info(f"Op-name mode: selected operator {self.op_name}")
            else:
                raise ValueError(f"Operator {self.op_name} not found in dataset. Available operators: {list(self.all_operators.keys())[:10]}...")
        # Single-test mode: randomly pick 1 operator
        elif self.single_test:
            import random
            items = list(self.all_operators.items())
            chosen = random.choice(items)
            self.all_operators = {chosen[0]: chosen[1]}
            logger.info(f"Single-test mode: selected operator {chosen[0]}")
        # Debug mode: limit to first 8 operators
        elif self.debug:
            self.all_operators = dict(list(self.all_operators.items())[:8])

        total_ops = len(self.all_operators)
        logger.info(f"Initialized {total_ops} operators")
    
    def get_remaining_operators(self) -> Dict[str, APIInfo]:
        """Get operators that have not yet passed"""
        remaining = {}
        for op_name, api_info in self.all_operators.items():
            if op_name not in self.passed_operators:
                remaining[op_name] = api_info
        return remaining
    
    def create_ut_generate_args(self, torch_op_name: str, torch_op_func_or_namespace: str, impl_info: APIInfo) -> TestFuncGenerateArgs:
        """Create test generation arguments."""
        kernel_name = torch_op_name.split('.')[-1]
        return TestFuncGenerateArgs(
            kernel_name=kernel_name,
            operators=impl_info.schemas,
            test_func_name=f"test_accuracy_{torch_op_func_or_namespace}::{kernel_name}",
            ops_namespace=torch_op_func_or_namespace,
        )
    
    def store_generated_code(self, op_name: str, round_idx: int, code: str) -> None:
        # Store the generated code
        if op_name not in self.generated_codes:
            self.generated_codes[op_name] = {}
        self.generated_codes[op_name][round_idx] = code
    
    def get_previous_verify_result(self, op_name: str, round_idx: int) -> Optional[Dict]:
        """Get verify result from previous round (from memory)."""
        if op_name in self.verify_results and round_idx in self.verify_results[op_name]:
            return self.verify_results[op_name][round_idx]
        return None
    
    def store_verify_result(self, op_name: str, round_idx: int, test_results: Dict) -> None:
        """Store verify result to memory."""
        if op_name not in self.verify_results:
            self.verify_results[op_name] = {}
        self.verify_results[op_name][round_idx] = test_results

    def generate_round(self, round_idx: int, remaining_operators: Dict[str, Dict[str, APIInfo]]) -> Path:

        """Generate tests for remaining operators in this round."""
        round_dir = self.output_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_idx}: Generating tests")
        if self.reflection and round_idx > 0:
            logger.info(f"Reflection enabled: Using previous round's verify results as feedback")
        logger.info(f"{'='*60}")
        
        # Prepare generation arguments
        gen_args = []
        api_names = []
        for op_name, api_info in remaining_operators.items():
            # op_name format: "aten::add"
            namespace, kernel_name = op_name.split("::", 1)
            file_name = f"test_accuracy_{op_name}.py"

            if Path(round_dir / file_name).exists():
                logger.info(f"Skipping already existing test for {op_name}")
                with open(round_dir / file_name, "r") as f:
                    code = f.read()
                self.store_generated_code(op_name, round_idx, code)
                continue

            # Choose impl_info based on test_type
            if self.test_type == "triton":
                impl_info_arg = self.adapter.get_impl_info(kernel_name)
            else:  # accuracy or performance
                impl_info_arg = api_info

            gen_arg = self.create_generate_args(
                torch_op_name=kernel_name,
                torch_op_func_or_namespace=namespace,
                impl_info=impl_info_arg
            )
            gen_arg.sample_id = round_idx

            # Add reflection: use previous round's verify results as feedback
            if self.reflection and round_idx > 0:
                prev_verify_result = self.get_previous_verify_result(op_name, round_idx - 1)
                if prev_verify_result:
                    from sandbox.utils.accuracy_utils import VerifyResult
                    # Extract failed test cases
                    failed_cases = prev_verify_result
                    if failed_cases:
                        # Take first few failures to avoid context overflow
                        sample_failures = failed_cases
                        verify_result_obj = VerifyResult(
                            **sample_failures
                        )
                        gen_arg.check_result = verify_result_obj
                        gen_arg.old_code = self.generated_codes[op_name][round_idx - 1]
                        assert gen_arg.check_result.code == gen_arg.old_code, "Mismatch between stored code and verify result code"
                        logger.info(f"Added reflection for {op_name}: {len(failed_cases)} failures from round {round_idx - 1}")

                    # Load previous generated code if available
                    if op_name in self.generated_codes and (round_idx - 1) in self.generated_codes[op_name]:
                        gen_arg.old_code = self.generated_codes[op_name][round_idx - 1]

            gen_args.append(gen_arg)
            api_names.append(op_name)
        
        if not gen_args:
            logger.info("No operators to generate")
            return round_dir
        
        logger.info(f"Generating {len(gen_args)} tests...")

        # Generate tests
        self.gen_config.sample_id = round_idx
        # Create Generator based on test_type; triton type needs PromptBuilder
        if self.test_type == "triton":
            generator = GENERATOR[self.test_type](self.gen_config, prompt_builder=self.prompt_builder)
        else:
            generator = GENERATOR[self.test_type](self.gen_config)
        generated_codes = generator(gen_args)
        
        # Process and save the generated codes
        generation_results = []
        saved_count = 0
        
        for idx, (generated_code, name, sample_id) in enumerate(generated_codes):
            full_name = api_names[idx]

            result_entry = {
                "operator": full_name,
                "test_file_name": f"{full_name}.py",
                "success": False,
                "error": None,
                "code_length": 0,
            }

            if generated_code and isinstance(generated_code, str) and len(generated_code.strip()) > 0:
                try:
                    test_file = round_dir / f"{full_name}.py"
                    with open(test_file, "w") as f:
                        f.write(generated_code)
                    
                    result_entry["success"] = True
                    result_entry["code_length"] = len(generated_code)
                    saved_count += 1
                    self.store_generated_code(full_name, round_idx, generated_code)
                    
                except Exception as e:
                    result_entry["error"] = str(e)
                    logger.error(f"Failed to save {name}: {e}")
            else:
                result_entry["error"] = "Empty or invalid generated code"
            
            generation_results.append(result_entry)
        
        round_summary_path = round_dir / "generation_summary.json"
        if round_summary_path.exists():
            exist_summary = json.load(open(round_summary_path, "r"))
            generation_results = exist_summary.get("results", []) + generation_results
            # Remove duplicates based on operator name
            seen_ops = set()
            unique_results = []
            for res in generation_results:
                if res["operator"] not in seen_ops:
                    unique_results.append(res)
                    seen_ops.add(res["operator"])
            generation_results = unique_results

        # Calculate statistics
        total = len(generation_results)
        successful = sum(1 for r in generation_results if r["success"])
        failed = total - successful
        
        # Create generation summary for this round
        generation_summary = {
            "round": round_idx,
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_apis": total,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total if total > 0 else 0,
            },
            "results": generation_results,
            "failed_operators": [r["operator"] for r in generation_results if not r["success"]],
        }
        
        # Save round generation summary
        with open(round_summary_path, "w") as f:
            json.dump(generation_summary, f, indent=2, ensure_ascii=False)
        
        # Store in overall summaries
        self.generation_summaries.append(generation_summary)
        
        # Log summary
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_idx} Generation Summary:")
        logger.info(f"Total APIs: {total}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {successful / total * 100:.2f}%" if total > 0 else "0%")
        logger.info(f"Summary saved to: {round_summary_path}")
        logger.info(f"{'='*60}\n")
        
        return round_dir
    
    def create_triton_kernel_verify_args(
            self,
            kernel_path: Path,
            test_path: Path,
            op_name: str,
            add_namespace_triton: bool = False
        ) -> VerifyRequest:
        """Create verification request. op_name format: aten::add"""
        test_func = None
        if test_path.is_file() and test_path.exists():
            with open(test_path, "r") as f:
                test_func = f.read()
        assert kernel_path.exists(), f"Kernel file does not exist: {kernel_path}"
        with open(kernel_path, "r") as f:
            kernel_code = f.read()
        return VerifyRequest(
            source=[Source(
                source=kernel_code,
                function_name=op_name, 
                namespace="triton" if add_namespace_triton else ""
            )],
            test_func=[test_func] if test_func else None,
        )
    
    def create_acc_test_verify_args(self, test_path: Path, op_name: str) -> VerifyRequest:
        """Create verification request. op_name format: aten::add"""
        with open(test_path, "r") as f:
            test_func = f.read()
        return VerifyRequest(
            source=[Source(
                source=mock_triton_code,
                function_name=op_name
            )],
            test_func=[test_func],
        )
    
    def verify_round(self, round_idx: int, round_dir: Path, remaining_operators: Dict[str, APIInfo]) -> Set[str]:
        """Verify tests for this round and return newly passed operators."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Round {round_idx}: Verifying tests")
        logger.info(f"{'='*60}")
        
        # Update verifier config
        # self.verify_config.run_name = f"pass_at_k_round_{round_idx}_{today()}"
        self.verify_config.sample_id = round_idx
        
        verifier = Verifier(self.verify_config)
        if self.custom_test_modules:
            mode = "performance" if self.test_type == "performance" else "accuracy"
            verifier.set_modules(
                modules=self.custom_test_modules,
                mode=mode
            )
        
        # Prepare verification requests
        verify_requests = []
        op_names = []
        exist_newly_passed = set()
        for op_name, api_info in remaining_operators.items():
            namespace, kernel_name = op_name.split("::", 1)
            kernel_file_name = f"{op_name}.py"
            kernel_path = round_dir / kernel_file_name

            if not kernel_path.exists():
                logger.warning(f"Missing kernel file: {kernel_path}")
                continue

            if self.test_type == "triton":
                test_file_path = Path("")
            else:
                test_file_path = round_dir / f"test_accuracy_{op_name}.py"

            if not test_file_path.exists() and self.test_type != "triton":
                logger.warning(f"Missing test file: {test_file_path}")
                continue
            verify_result_path = Path(test_file_path).parent.parent / "verification" / f"log_{round_idx}" / "result.json"
            if verify_result_path.exists():
                logger.info(f"Skipping already verified test for {op_name}")
                # check previous result
                with open(verify_result_path, "r") as f:
                    reports = json.load(f)
                # Store verify result to memory
                current_report = {}
                for report in reports:
                    if report["op_name"] == op_name:
                        current_report = report
                        self.store_verify_result(op_name, round_idx, current_report)
                if current_report.get("success", False):
                    logger.info(f"Test already passed for {op_name}")
                    exist_newly_passed.add(op_name)
                    self.passed_operators.add(op_name)
                continue

            verify_req = self.create_triton_kernel_verify_args(
                kernel_path,
                test_file_path,
                op_name, 
                add_namespace_triton=not op_name.startswith("aten::")
            )
            verify_requests.append(verify_req)
            op_names.append(op_name)
        
        if not verify_requests:
            logger.info("No tests to verify this time.")
            return exist_newly_passed
        
        logger.info(f"Verifying {len(verify_requests)} tests...")
        
        # Run verification
        _, results = verifier.only_verify(
            name_source_map=verify_requests,
            device_count=self.device_count,
        )
        
        # Collect newly passed operators and store verify results
        newly_passed = set()
        for result, op_name in zip(results, op_names):
            # Load and store the detailed test report from file
            kernel_name = op_name.split("::")[-1]
            # verify_result_path = self.output_dir / "verification" / f"log_{round_idx}" / f"test_report_{kernel_name}.json"
            verify_result_path = self.output_dir / "verification" / f"log_{round_idx}" / "result.json"
            if verify_result_path.exists():
                try:
                    with open(verify_result_path, "r") as f:
                        test_reports = json.load(f)
                    current_report = {}
                    for report in test_reports:
                        if report["op_name"] == op_name:
                            current_report = report
                    self.store_verify_result(op_name, round_idx, current_report)
                    logger.debug(f"Stored verify result for {op_name} round {round_idx}")
                except Exception as e:
                    logger.warning(f"Failed to load test report for {op_name}: {e}")
            
            if result.success:
                newly_passed.add(result.op_name)
        
        newly_passed.update(exist_newly_passed)
        
        logger.info(f"Passed: {len(newly_passed)}/{len(verify_requests)} tests")
        return newly_passed
    
    def run_pass_at_k(self, max_rounds: int = 10) -> None:
        """Run pass@k testing for multiple rounds."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting Pass@{max_rounds} Testing")
        logger.info(f"{'='*60}\n")

        total_operators = len(self.all_operators)

        # Track when each operator first passed
        self.first_pass_round: Dict[str, int] = {}

        for round_idx in range(max_rounds):
            # Get remaining operators
            remaining = self.get_remaining_operators()
            remaining_count = len(remaining)
            
            if remaining_count == 0:
                logger.info(f"\n{'='*60}")
                logger.info(f"All operators passed! Stopping at round {round_idx}")
                logger.info(f"{'='*60}\n")
                break
            
            logger.info(f"\nRound {round_idx}: {remaining_count}/{total_operators} operators remaining")
            
            # Generate tests
            round_dir = self.generate_round(round_idx, remaining)
            
            # Verify tests
            newly_passed = self.verify_round(round_idx, round_dir, remaining)
            
            # Update passed operators and track first pass round
            for op_name in newly_passed:
                if op_name not in self.passed_operators:
                    self.first_pass_round[op_name] = round_idx
            
            self.passed_operators.update(newly_passed)
            
            # Record round results
            round_result = {
                "round": round_idx,
                "remaining_before": remaining_count,
                "newly_passed": len(newly_passed),
                "newly_passed_operators": sorted(list(newly_passed)),
                "total_passed": len(self.passed_operators),
                "pass_rate": len(self.passed_operators) / total_operators,
            }
            self.results_by_round.append(round_result)
            
            # Save intermediate results
            self.save_results()
            
            logger.info(f"\nRound {round_idx} Summary:")
            logger.info(f"  Newly passed: {len(newly_passed)}")
            logger.info(f"  Total passed: {len(self.passed_operators)}/{total_operators}")
            logger.info(f"  Pass rate: {round_result['pass_rate']:.2%}")
        
        # Anti-hack: final check on all passed operators
        if self.passed_operators:
            self.anti_hack_final_check()

        # Final summary
        self.print_final_summary(total_operators, max_rounds)
    
    def anti_hack_final_check(self) -> None:
        """Run anti-hack checks on all passed operators after Pass@K completes.

        Does NOT modify self.passed_operators. Saves a separate
        pass_at_k_results_antihack.json with anti-hack results.
        """
        # Prepare operators dict for AntiHackRunner
        operators = {}
        for op_name in list(self.passed_operators):
            round_idx = self.first_pass_round.get(op_name)
            if round_idx is None:
                continue

            codes = self.generated_codes.get(op_name, {})
            kernel_code = codes.get(round_idx, "")
            if not kernel_code:
                continue

            round_dir = self.output_dir / f"round_{round_idx}"
            kernel_path = round_dir / f"{op_name}.py"
            test_file_path = None if self.test_type == "triton" else round_dir / f"test_accuracy_{op_name}.py"

            operators[op_name] = {
                "kernel_code": kernel_code,
                "kernel_path": kernel_path,
                "test_file_path": test_file_path,
            }

        # Run anti-hack checks
        from sandbox.anti_hack_runner import AntiHackRunner
        runner = AntiHackRunner(
            self.dataset, self.verify_config,
            custom_test_modules=self.custom_test_modules,
        )
        hack_results = runner.batch_check(operators)
        runner.save_report(
            hack_results=hack_results,
            total_operators=len(self.all_operators),
            passed_operators=self.passed_operators,
            output_path=self.output_dir / "pass_at_k_results_antihack.json",
        )

    def save_results(self) -> None:
        """Save current results to JSON."""
        results_file = self.output_dir / "pass_at_k_results.json"
        
        # Prepare detailed results with codes
        operator_details = {}
        for op_name in self.generated_codes.keys():
            namespace, kernel_name = op_name.rsplit("::", 1) if "::" in op_name else ("", op_name)
            operator_details[op_name] = {
                "passed": op_name in self.passed_operators,
                "namespace": namespace,
                "kernel_name": kernel_name,
                "generated_codes_by_round": self.generated_codes[op_name],
                "first_pass_round": self.first_pass_round.get(op_name, None),
                "total_attempts": len(self.generated_codes[op_name])
            }
        
        results = {
            "total_operators": len(self.all_operators),
            "total_passed": len(self.passed_operators),
            "final_pass_rate": len(self.passed_operators) / len(self.all_operators) if len(self.all_operators) > 0 else 0,
            "rounds_summary": self.results_by_round,
            "generation_summaries": self.generation_summaries,
            "passed_operators": sorted(list(self.passed_operators)),
            "operator_details": operator_details,
        }
        
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to: {results_file}")
    
    def print_final_summary(self, total_operators: int, max_rounds: int) -> None:
        """Print final summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Pass@K Testing Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total operators: {total_operators}")
        logger.info(f"Total passed: {len(self.passed_operators)}")
        logger.info(f"Final pass rate: {len(self.passed_operators) / total_operators:.2%}")
        
        logger.info(f"\nGeneration Statistics:")
        for gen_summary in self.generation_summaries:
            stats = gen_summary["statistics"]
            logger.info(f"  Round {gen_summary['round']}:")
            logger.info(f"    Generated: {stats['successful']}/{stats['total_apis']}")
            logger.info(f"    Success Rate: {stats['success_rate']:.2%}")
        
        logger.info(f"\nVerification Statistics (Pass@K):")
        for i, round_result in enumerate(self.results_by_round, 1):
            logger.info(f"  Pass@{i}: {round_result['pass_rate']:.2%} ({round_result['total_passed']}/{total_operators})")
        
        logger.info(f"{'='*60}\n")


def main():
    # Auto-detect default dataset based on device type
    # NVIDIA: KernelGenBench (210 ops); Other chips: KernelGenBench-aten (110 ops)
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from runtime import get_device_type
        _device_type = get_device_type()
    except Exception:
        _device_type = "nvidia"
    _DEFAULT_DATASET = "KernelGenBench" if _device_type == "nvidia" else "KernelGenBench-aten"

    parser = argparse.ArgumentParser(description="Test Pass@K for operator generation")

    parser.add_argument("--name", type=str, default="aten", help="Namespace to test (default: aten)")
    parser.add_argument("--acc-test-func-path", type=str, default="", help="Path to the accuracy test function directory")
    parser.add_argument("--benchmark-func-path", type=str, default="", help="Path to the performance test function directory")
    parser.add_argument("--dataset", type=str, default=_DEFAULT_DATASET, help=f"Dataset version to use (default: {_DEFAULT_DATASET})", choices=["KernelGenBench", "KernelGenBench-aten", "KernelGenBench-vllm", "KernelGenBench-cublas", "KernelGenBench-nocublas", "MmShapeBench"])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "pass_at_k", help="Output directory")
    parser.add_argument("--resume-from", type=Path, help="Resume from existing checkpoint directory")
    parser.add_argument("--test-type", type=str, default="triton", choices=["accuracy", "performance", "triton"])
    parser.add_argument("--max-rounds", type=int, default=10, help="Maximum number of rounds (default: 10)")
    parser.add_argument("--device-count", type=int, default=8, help="Number of devices for testing")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout for each test")
    
    # Generation config
    parser.add_argument(
        "--api-format", "--server-type",
        dest="server_type",
        choices=["openai", "anthropic"],
        default="openai",
        help="API wire format; --server-type is kept as a backward-compatible alias",
    )
    parser.add_argument("--model-name", type=str, default="gpt-4o-mini")
    parser.add_argument("--base-url", type=str, default=None, help="API base URL for either supported API format")
    parser.add_argument("--api-key", type=str, default=None, help="API key (overrides OPENAI_API_KEY / ANTHROPIC_API_KEY env var)")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--num-workers", type=int, default=150)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode (8 operators)")
    parser.add_argument("--single-test", action="store_true", help="Single-test mode: randomly pick 1 operator")
    parser.add_argument("--op-name", type=str, default=None, help="Specify a single operator to test (e.g., vllm13::fused_add_rms_norm)")
    parser.add_argument("--reflection", action="store_true", help="Enable reflection: use previous round's verify results as feedback for next generation")
    parser.add_argument("--custom-test-modules", type=str, nargs="+", default=None, help="Custom test module paths or directories (e.g., src/kernelgenbench/accuracy/test_custom.py or src/kernelgenbench/accuracy/)")

    args = parser.parse_args()

    check_args_validity(args)
    
    # Determine output directory and resume mode
    if args.resume_from:
        output_dir = args.resume_from
        logger.info(f"Resuming from: {output_dir}")
    else:
        postfix = ""
        if args.op_name:
            # Clean up operator name for use in directory name
            clean_name = args.op_name.replace("::", "_").replace("/", "_")
            postfix += f"_{clean_name}"
        elif args.single_test:
            postfix += "_single"
        elif args.debug:
            postfix += "_debug"
        if args.reflection:
            postfix += "_reflection"
        run_name = f"pass_at_{args.max_rounds}_{args.model_name}_{args.test_type}{postfix}_{today()}"

        # Adjust base directory based on test_type
        if args.test_type == "accuracy":
            base_dir = args.output_dir.parent / f"{args.output_dir.name}_accuracy"
        else:
            base_dir = args.output_dir

        output_dir = base_dir / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # save args to output dir
    args_file = output_dir / "args.json"
    with open(args_file, "w") as f:
        args_dict = vars(args).copy()
        if args_dict.get("api_key"):
            args_dict["api_key"] = "<redacted>"
        # Convert Path objects to strings for JSON serialization
        for key, value in args_dict.items():
            if isinstance(value, Path):
                args_dict[key] = str(value)
        json.dump(args_dict, f, indent=2)

    run_name = output_dir.name
    
    # Create generation config
    # Set only the environment variable for the selected wire format. The
    # inference client reads it when each request is created, so CLI values work
    # even though this module imports the generator before parsing arguments.
    if args.api_key:
        key_env = "ANTHROPIC_API_KEY" if args.server_type == "anthropic" else "OPENAI_API_KEY"
        os.environ[key_env] = args.api_key

    gen_config = GenerationConfig(
        run_name="",
        server_type=args.server_type,
        model_name=args.model_name,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        num_samples=1,
        verbose=False,
        run_dir=str(output_dir),
        log_prompt=True,
    )
    
    # Create verification config
    verify_config = VerifyConfig(
        run_name="",
        test_type=args.test_type,
        run_dir=str(output_dir / "verification"),
        store_type="local",
        strict_check=True,
        seed=42,
        sample_id=0,
        save_log=True,
        acc_timeout=args.timeout,
        perf_timeout=args.timeout,
    )
    
    # Create tester and run
    tester = PassAtKTester(
        output_dir=output_dir,
        test_type=args.test_type,
        acc_test_func_path=args.acc_test_func_path,
        bench_test_func_path=args.benchmark_func_path,
        dataset=args.dataset,
        custom_test_modules=args.custom_test_modules,
        gen_config=gen_config,
        verify_config=verify_config,
        device_count=args.device_count,
        debug=args.debug,
        single_test=args.single_test,
        op_name=args.op_name,
        reflection=args.reflection,
    )
    
    tester.initialize_operators(args.name)
    tester.run_pass_at_k(max_rounds=args.max_rounds)


if __name__ == "__main__":
    main()
