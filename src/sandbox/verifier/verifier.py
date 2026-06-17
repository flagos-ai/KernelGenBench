from .utils import expand_params, save_benchmark_result, add_register_decorator
from sandbox.utils.accuracy_utils import VerifyResult
from sandbox.register_scanner import auto_register_module
from sandbox.verifier.test_parametrize import get_funcs_by_label, _label_registry, get_params
from fastapi.encoders import jsonable_encoder
import os
DISPATCH_TORCH_LIB = os.environ.get("DISPATCH_TORCH_LIB", "1") == "1"
from tqdm import tqdm
import traceback
import tempfile
from typing import Callable, Union, List, Optional, Any
import json
import torch
import multiprocessing as mp
import numpy as np
import importlib
from pydantic import BaseModel
from copy import deepcopy
from dataclasses import dataclass, asdict
class BenchmarkResult:
    """Stub: legacy benchmark result type, no longer used."""
    pass
from sandbox.utils.accuracy_utils import CustomBenchmarkResult


from kernelgenbench.dataset import is_pytorch_op, IMPL_INFO
def get_visible_devices_env():
    return "CUDA_VISIBLE_DEVICES"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    pass

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class VerifyConfig:
    run_name: str
    test_type: str = "accuracy" # accuracy, performance, both
    run_dir: str = os.path.join(REPO_TOP_DIR, "runs")
    store_type: str = "local"
    strict_check: bool = False
    seed: int = 42
    sample_id: int = 0
    save_log: bool = True
    acc_timeout: int = 300    # seconds
    perf_timeout: int = 600    # seconds
    manage_device_visibility: bool = True  # Whether to set device visibility env var
    anti_hack: bool = True  # Enable anti-hack Layer 2/3 runtime checks

@dataclass
class Source:
    source: str
    function_name: str
    namespace: str = ""

@dataclass
class VerifyRequest:
    source: List[Source]
    test_func: List[str | None] | None = None
    test_func_mark: str | None = None

import logging
from .utils import CustomRichHandler, generate_speedup_html
from rich.console import Console
console = Console()


logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[CustomRichHandler(
        rich_tracebacks=True,
        show_path=True, 
        show_level=True, 
        show_time=True 
    )]
)

logger = logging.getLogger(__name__)

mp.set_start_method("spawn", force=True)

def default_converter(o):
    if hasattr(o, "item"):
        return o.item()
    if isinstance(o, (np.generic,)):
        return o.item()
    if "torch" in str(type(o)):
        return str(o)
    if isinstance(o, BaseModel):
        return o.model_dump()
    if isinstance(o, Callable):
        return o.__name__
    if isinstance(o, complex):
        return str(o)
    
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

from typing import List, Dict, Tuple

# USE_OPS = True

def get_name_list_from_code_dir(code_dir: str) -> List[str]:
    name_list = []
    for file in os.listdir(code_dir):
        if file.endswith(".py") and file.startswith("problem_"):
            name = file[len("problem_"):].split("_sample_")[0]
            name_list.append(name)
    return name_list


class Verifier:
    def __init__(self, config: VerifyConfig):
        self.config = config
        self._running_config = deepcopy(config)
        self.accuracy_modules = []
        self.perf_modules = []
        self.external_modules_set = False
        self._setup_sandbox()

    def _setup_sandbox(self):
        """Apply lightweight sandbox protections before any verification.

        Env vars are inherited by subprocesses.
        CUDA protector applies in-process immediately.
        Import hook is intentionally NOT enabled here (too heavy for daily use).
        For S-Level competition sandbox, use competition_evaluator.py.
        """
        import os

        # 1. Disable triton autotune / torch compile caches (inherited by subprocesses)
        os.environ.setdefault("TRITON_DISABLE_AUTOTUNE", "1")
        os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/triton_cache")
        os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
        os.environ.setdefault("CUDA_CACHE_DISABLE", "1")

        # 2. CUDA layer protection (disable CUDA Graph, TF32, reset state)
        try:
            from sandbox.cuda_protector import CUDALayerProtector
            self._cuda_protector = CUDALayerProtector()
            self._cuda_protector.setup()
        except Exception:
            self._cuda_protector = None

    def set_modules(self, modules: list, mode: str = "accuracy"):
        assert mode in ["accuracy", "performance"], f"mode must be accuracy or performance, got {mode}"
        if mode == "accuracy":
            self.accuracy_modules = modules
        if mode == "performance":
            self.perf_modules = modules
        self.external_modules_set = True


    def _import_module_or_path(self, module_or_path: str):
        """
        Dynamically import a module or file path.
        - If it is a directory path, import all .py files in the directory.
        - If it is a file path (ending in .py or containing a path separator), load from file.
        - Otherwise treat as a module name and import.
        """
        import sys
        import importlib.util

        # Check if it is a directory
        if os.path.isdir(module_or_path):
            logger.info(f"Loading all test modules from directory: {module_or_path}")
            for filename in sorted(os.listdir(module_or_path)):
                if filename.endswith('.py') and not filename.startswith('_'):
                    file_path = os.path.join(module_or_path, filename)
                    self._import_module_or_path(file_path)
            return

        # Check if it is a file path
        if os.path.exists(module_or_path) or module_or_path.endswith('.py') or os.path.sep in module_or_path:
            # Handle as file path
            if not os.path.exists(module_or_path):
                raise FileNotFoundError(f"Module file not found: {module_or_path}")
            
            # Generate module name (extracted from filename)
            module_name = os.path.splitext(os.path.basename(module_or_path))[0]
            
            # Use importlib.util to load from file
            spec = importlib.util.spec_from_file_location(module_name, module_or_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load module from {module_or_path}")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            logger.info(f"Loaded module from file: {module_or_path}")
        else:
            # Handle as module name
            importlib.import_module(module_or_path)
            logger.info(f"Imported module: {module_or_path}")


    def import_tests(self, mode: str = "accuracy"):
        import os
        if os.environ.get("KERNELGENBENCH_SKIP_BOTH_TEST", "0") == "1" and self.external_modules_set is False:
            logger.info("Skipping both accuracy and performance test imports due to KERNELGENBENCH_SKIP_BOTH_TEST=1")
            return
        if not self.accuracy_modules:
            from kernelgenbench import accuracy_modules
            self.accuracy_modules = accuracy_modules
        if mode == "accuracy":
            modules = self.accuracy_modules
        elif mode == "performance":
            modules = self.perf_modules
        elif mode == "both":
            modules = self.accuracy_modules + self.perf_modules
        for module in modules:
            self._import_module_or_path(module)


    def _init_test_func(self):
        self.import_tests(self._running_config.test_type)

    def _update_config(self, config: VerifyConfig):
        self.config = config
        self._running_config = deepcopy(config)

    def _summary(self, result: List):
        # check if result.json exists and load it
        log_dir = os.path.join(
            self._running_config.run_dir, 
            self._running_config.run_name, 
            f"log_{self._running_config.sample_id}"
        )
        if os.path.exists(os.path.join(log_dir, "result.json")):
            with open(os.path.join(log_dir, "result.json"), "r") as f:
                existing_result = json.load(f)
                existing_result = [VerifyResult(**r) for r in existing_result]
            # merge existing result with new result, avoid duplicate by op_name
            existing_op_names = set(r.op_name for r in existing_result)
            for r in result:
                if r.op_name not in existing_op_names:
                    existing_result.append(r)
            result = existing_result
        summary = {
            "total": len(result),
            "passed": sum(1 for r in result if r.success is True),
            "failed": sum(1 for r in result if r.success is False),
        }
        no_test = sum(1 for r in result if r.success is None)
        no_test_list = [r.op_name for r in result if r.success is None]
        info = {
            "info": f"there are {no_test} tests without results: {no_test_list}", 
            "no_test_list": no_test_list, 
            "failed_name_list": [r.op_name for r in result if r.success is False]
        }
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2, default=default_converter)
        with open(os.path.join(log_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2, default=default_converter)
        logger.info(f"failed {info}")
        return summary, info

    def _auto_load_baseline(self, op_name: str) -> Optional[str]:
        """Auto-load baseline implementation.

        Attempt order:
        1. By naming convention: cublas::gemm -> baseline/cublas/gemm.py
        2. custom directory: baseline/custom/<op_name>.py
        3. example directory (for testing): baseline/example/baseline_<op_name>.py
        """
        from pathlib import Path

        baseline_root = Path(REPO_TOP_DIR).parent / "kernelgenbench" / "dataset" / "baseline"

        # Attempt 1: parse naming convention (category::opname)
        if "::" in op_name:
            parts = op_name.split("::", 1)
            category, name = parts[0], parts[1]
            baseline_path = baseline_root / category / f"{name}.py"
            if baseline_path.exists():
                logger.info(f"Auto-loading baseline from {baseline_path}")
                return baseline_path.read_text()

        # Attempt 2: custom directory
        baseline_path = baseline_root / "custom" / f"{op_name}.py"
        if baseline_path.exists():
            logger.info(f"Auto-loading baseline from {baseline_path}")
            return baseline_path.read_text()

        # Attempt 3: example directory (for testing)
        # For non_torch_prelu, try baseline_prelu.py
        if op_name.startswith("non_torch_"):
            simple_name = op_name.replace("non_torch_", "")
            baseline_path = baseline_root / "example" / f"baseline_{simple_name}.py"
            if baseline_path.exists():
                logger.info(f"Auto-loading baseline from {baseline_path}")
                return baseline_path.read_text()

        logger.warning(f"Baseline not found for {op_name}")
        return None

    def _check_code(self, code: str, name: str, namespace: str = None):
        if not code:
            return None
        def ensure_import_torch(code: str) -> str:
            package_list = ["torch", "triton"]
            # Check whether the string contains the package name
            for package in package_list:
                if package in code:
                    # If the package is present, check whether "import <package>" already exists
                    if f"import {package}" not in code:
                        # If not, prepend the import statement
                        code = f"import {package}\n" + code
            if "tl." in code:
                if "import triton.language as tl" not in code:
                    code = "import triton.language as tl\n" + code
            return code
        
        # Save the original full name (used to determine whether it is a PyTorch operator)
        original_name = name

        name = name.split("::")[-1] if "::" in name else name
        name = name.split(".")[-1]
        compile(code, name, "exec")

        # Step 2.1.1: add operator type check
        # Use the original name (with namespace) to avoid defaulting to aten namespace
        _is_torch_op = is_pytorch_op(original_name)

        if _is_torch_op:
            # PyTorch operator: check all overload variants
            impl_info = IMPL_INFO.get(name)
            ops = [op for op, _ in impl_info]
        else:
            # Non-PyTorch operator: only check the main function name
            ops = [name]

        # Uniformly check function definitions
        for op in ops:
            op_func_name = op.replace(".", "_")
            if f"def {op_func_name}(" not in code:
                logger.error(f"no func {op_func_name} in code \n{code}")
                raise ValueError(f"no func {op_func_name} in code, must include def {op_func_name}")
        
        # check package import
        code = ensure_import_torch(code)
        if "@register" not in code:
            code = "from kernelgenbench import register\n" + code

            # Use the passed-in namespace directly without remapping
            actual_namespace = namespace

            for op in ops:
                code = add_register_decorator(code, op, actual_namespace, api=name)
        if not os.path.isfile(code):
            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
                tmp.write(code)
                triton_code_paths = tmp.name
                logger.info(f"save triton code to {triton_code_paths}")
        auto_register_module(triton_code_paths)
        return triton_code_paths

    def only_verify(
        self, 
        name_source_map: List[VerifyRequest],
        test_type: str = "accuracy", # accuracy or performance
        device_count: int = 1, 
    ) -> Tuple[Dict, List[VerifyResult]]:
        self._running_config.test_type = test_type
        results = self.verify(name_source_maps=name_source_map, device_count=device_count)
        summary, info = self._summary(results)
        return summary, results


    def _verify_with_one_device(self, task_collections: List[VerifyRequest], device_id: int, result_queue: mp.Queue):
        # Only set device visibility if manage_device_visibility is enabled
        # When running under VerifierServer, this should be False as the server manages visibility
        if self.config.manage_device_visibility:
            os.environ[get_visible_devices_env()] = str(device_id)
        ctx = mp.get_context("spawn")
        for verifyrequest in task_collections:
            p  = ctx.Process(target=self._verify, kwargs={
                "verifyrequest": verifyrequest, 
                "result_queue": result_queue,
            })
            p.start()
            p.join(timeout=self.config.acc_timeout)
            if p.is_alive():
                logger.error(f"TimeoutError: Test for {verifyrequest.source[0].function_name} timed out after {self.config.acc_timeout}s.")
                p.terminate()
                p.join()
                result_queue.put(VerifyResult(
                    op_name=verifyrequest.source[0].function_name,
                    success=False,
                    traceback=f"TimeoutError: Test timed out after {self.config.acc_timeout} seconds.",
                    code=verifyrequest.source[0].source if verifyrequest.source else None,
                    test_func=verifyrequest.test_func[0] if verifyrequest.test_func else None,
                ))
            elif p.exitcode != 0:
                logger.error(f"Process for {verifyrequest.source[0].function_name} exited with code {p.exitcode}")
                result_queue.put(VerifyResult(
                    op_name=verifyrequest.source[0].function_name,
                    success=False,
                    # TODO: catch actual traceback from process, stdout/err
                    traceback=f"Process exited with code {p.exitcode}",
                    code=verifyrequest.source[0].source if verifyrequest.source else None,
                    test_func=verifyrequest.test_func[0] if verifyrequest.test_func else None,
                ))

    def verify(
        self,
        name_source_maps: List[VerifyRequest],
        device_count: int = 1
    ) -> List[VerifyResult]:
        """

        """
        check_result = []
        # if device_count > 1 and len(name_source_maps) > 1:
        total_tasks = len(name_source_maps)
        task_chunks = [[] for _ in range(device_count)]
        for i, name_source_map in enumerate(name_source_maps):
            task_chunks[i % device_count].append(name_source_map)
        result_queue = mp.Queue()

        # One process per GPU
        processes: List[mp.Process] = []
        for i, chunk in enumerate(task_chunks):
            p = mp.Process(target=self._verify_with_one_device, args=(chunk, i, result_queue))
            p.start()
            processes.append(p)
        
        with tqdm(total=total_tasks, desc="All GPUs") as pbar:
            finished = 0
            total = 0
            success = 0
            while finished < total_tasks:
                result = result_queue.get()  # wait for child process message
                s = result.success == True
                success += s
                check_result.append(result)
                total += 1 if result.success is not None else 0
                finished += 1
                pbar.update(1)
                pbar.set_postfix(success=f"{success}/{total}")
        for p in processes:
            p.join()

        return check_result

    def run_tests(
        self, 
        name, 
        json_path: str=None, 
        max_failures: str | int = "all", 
        seed=42, 
        strict_check=False
    ) -> VerifyResult:
        set_seed(seed)
        # get api name from op_mark if it contains "::"
        report_name, name = name, name.split("::")[-1] if "::" in name else name
        total = 0
        failed = 0
        ret = None
        funcs = get_funcs_by_label(name)
        # When the operator name has a prefix (e.g. vllm13::rms_norm), filter out test functions
        # that do not belong to that prefix, to avoid matching the wrong test when operators with
        # the same name exist in different namespaces (e.g. aten and vllm13 both have rms_norm)
        if "::" in report_name:
            prefix = report_name.split("::")[0]  # e.g. "vllm13", "cublas"
            filtered = [(f, m) for f, m in funcs if prefix in f.__module__]
            if filtered:
                funcs = filtered
        print(funcs)
        results = []
        speedup = None
        first_failure_traceback = None  # Store first failure traceback for reporting
        first_func_combo = None
        first_normal_output = None  # Saved from accuracy test for Layer 2 optimization
        for func, mark in funcs:
            func_name = func.__name__
            if func is None:
                logger.info(f"Fail Test function {func_name} not found")
                console.print(f"[red][bold]Fail[/bold][/red] Test function {func_name} not found")
                return VerifyResult(op_name=report_name, success=False, traceback=f"Test function {func_name} not found")
            params = get_params(func_name, mark)
            for combo in ([{}] if not params else expand_params(params)):
                total += 1
                success = True
                tb_str = None
                speed = None  # Reset speed for each test case to avoid reusing previous value
                ret = None    # Reset ret for each test case to avoid wrong branch on failure
                if first_func_combo is None:
                    first_func_combo = (func, deepcopy(combo))
                try:
                    # recorded_params = jsonable_encoder(combo, custom_encoder={torch.dtype: str, Callable: lambda x: x.__name__})
                    # recorded_params = {k: default_converter(v) for k, v in combo.items()}
                    recorded_params = json.loads(json.dumps(combo, default=default_converter))
                except Exception as e:
                    print(combo)
                    raise e
                try:
                    ret = func(**combo)
                    if first_normal_output is None:
                        first_normal_output = deepcopy(ret) if ret is not None else None
                except Exception as e:
                    tb_str = traceback.format_exc()
                    success = False
                if ret is not None:
                    if isinstance(ret, list) and len(ret) > 0 and isinstance(ret[0], BenchmarkResult):
                        if speedup is None:
                            speedup = []
                        speed_save_path = json_path.replace(".json", "_speedup.txt") if json_path else None
                        # clear the previous saved speedup file
                        if speed_save_path and os.path.exists(speed_save_path):
                            os.remove(speed_save_path)
                        for r in ret:
                            if isinstance(r, BenchmarkResult):
                                save_benchmark_result(r, speed_save_path)
                            speed = json.loads(r.to_json()) if isinstance(r, BenchmarkResult) else r.model_dump()
                            speed["params"] = recorded_params
                            avg_speed = {
                                "ref_time": np.mean([s["latency_base"] for s in speed['result']]).item(),
                                "res_time": np.mean([s["latency"] for s in speed['result']]).item(),
                                "speedup": np.exp(np.mean(np.log([s["speedup"] for s in speed['result']]))).item(),
                                "params": speed['dtype'],
                            }
                            # speed['result'].append(avg_speed)
                            speedup.append(avg_speed)
                        # speedup = speed['result']
                    if isinstance(ret, CustomBenchmarkResult):
                        if speedup is None:
                            speedup = []
                        # Save CustomBenchmarkResult similar to BenchmarkResult
                        speed_save_path = json_path.replace(".json", "_speedup.txt") if json_path else None

                        speed = json.loads(ret.json())
                        speed["params"] = recorded_params
                        if speed_save_path:
                            save_benchmark_result(speed, speed_save_path)
                        speedup.append(speed)
                    results.append({
                        "params": combo,
                        "success": success,
                        "traceback": tb_str,
                        "speedup": speed,
                    })
                else:
                    results.append({
                        "params": combo,
                        "success": success,
                        "traceback": tb_str,
                    })
                if not success:
                    if first_failure_traceback is None:
                        first_failure_traceback = tb_str  # Save first failure traceback
                    if strict_check or ("Tensor-likes are not close" not in tb_str):
                        failed += 1
                    if max_failures != "all" and failed >= max_failures:
                        break
            # compute average speedup for all combos of this func and update it in speedup list, name params as avg
            if speedup is not None:
                if len(speedup) > 0:
                    avg_speed = {
                        "ref_time": np.mean([s['ref_time'] for s in speedup]).item(),
                        "res_time": np.mean([s['res_time'] for s in speedup]).item(),
                        "speedup": np.exp(np.mean(np.log([s['speedup'] for s in speedup]))).item(),
                        "params": "avg",
                    }
                    speedup.append(avg_speed)

            if not success:
                break
        if total == 0:
            logger.warning(f"[\033[91mFail\033[0m] No valid test cases found for {name}")
            return VerifyResult(
                op_name=report_name,
                success=None, 
                traceback=None,
                params=None, 
                info = {
                    "total": total,
                    "failed": failed,
                    "success": total - failed
                }
            )

        # Anti-hack Layer 2 & 3: only when config.anti_hack is enabled
        if self._running_config.anti_hack and failed == 0 and first_func_combo is not None:
            from sandbox.anti_hack import dual_execution_check_with_ref, gpu_profiling_check
            ah_func, ah_kwargs = first_func_combo
            hack_detected = False
            hack_reason = ""
            # Layer 2: optimized — reuse already-computed normal output from accuracy test
            if first_normal_output is not None:
                try:
                    is_hack, reason = dual_execution_check_with_ref(ah_func, ah_kwargs, first_normal_output)
                    if is_hack:
                        hack_detected, hack_reason = True, reason
                except Exception as e:
                    logger.debug(f"Anti-hack Layer2 skipped: {e}")
            else:
                from sandbox.anti_hack import dual_execution_check
                try:
                    is_hack, reason = dual_execution_check(ah_func, ah_kwargs)
                    if is_hack:
                        hack_detected, hack_reason = True, reason
                except Exception as e:
                    logger.debug(f"Anti-hack Layer2 skipped: {e}")
            if not hack_detected:
                try:
                    is_hack, reason = gpu_profiling_check(ah_func, ah_kwargs)
                    if is_hack:
                        hack_detected, hack_reason = True, reason
                except Exception as e:
                    logger.debug(f"Anti-hack Layer3 skipped: {e}")
            if hack_detected:
                failed = total
                tb_str = f"[Anti-hack] {hack_reason}"
                logger.warning(f"Anti-hack detected for {report_name}: {hack_reason}")

        log_flag = "[Fail]" if failed > 0 else "[Success]"
        flag = "[green]Success[/green]" if failed == 0 else "[red]Fail[/red]"
        if json_path:
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, default=default_converter)
            logger.info(f"{log_flag} Test results saved to {json_path}")
            console.print(f"{flag} Test results saved to {json_path}")

        return VerifyResult(
            op_name=report_name,
            success=failed == 0,
            traceback=first_failure_traceback,  # Use first failure traceback instead of last case's tb_str
            params=recorded_params,
            speedup=speedup,
            info={
                "total": total,
                "failed": failed,
                "success": total - failed
            }
        )

    def run_test(
        self, 
        op_names: str, 
        config: VerifyConfig, 
    ) -> VerifyResult:
        op_mark = op_names.split(".")[-1] if "." in op_names else op_names

        if config.save_log:
            log_dir = os.path.join(
                config.run_dir, 
                config.run_name, 
                f"log_{config.sample_id}"
            )
        else:
            log_dir = None

        # === Handle JSON save path ===
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            json_path = os.path.join(log_dir, f"test_report_{op_mark}.json")
            delete_after = False
        else:
            tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
            json_path = tmpfile.name
            tmpfile.close()
            delete_after = True

        verifyresult = self.run_tests(
            name=op_mark, 
            json_path=json_path, 
            max_failures="all",
            seed=config.seed, 
            strict_check=config.strict_check,
        )

        if delete_after:
            os.remove(json_path)
        return verifyresult
    
    def _verify(
        self,
        verifyrequest: VerifyRequest,
        result_queue: mp.Queue = None, 
        namespace: Union[str, List[str]] = None, 
    ) -> VerifyResult:
        source = [s.source for s in verifyrequest.source]
        namespace = [s.namespace for s in verifyrequest.source]
        function_name = [s.function_name for s in verifyrequest.source]
        function_name = [fn.split(".")[-1] for fn in function_name]
        test_func = verifyrequest.test_func
        test_func_mark = verifyrequest.test_func_mark
        read_file = lambda s: open(s, "r").read()
        source = [s if not s or not os.path.isfile(s) else read_file(s) for s in source]

        try:
            self._init_test_func()    # For external Triton kernel developers, there is no need to initialize kernelgenbench test functions; use this line if running as a benchmark
        except Exception as e:
            logger.error(f"Init test functions failed: {e}")
            raise e
        try:
            # Step 2.3: handle DISPATCH_TORCH_LIB environment variable
            _is_torch_op = is_pytorch_op(function_name[0])

            # Auto-load baseline (approach 1 + approach 4)
            if not _is_torch_op:
                # Check whether a baseline has already been provided
                has_baseline = any(ns == "baseline" for ns in namespace)

                if not has_baseline:
                    # Try to auto-load baseline
                    baseline_code = self._auto_load_baseline(function_name[0])
                    if baseline_code:
                        source.insert(0, baseline_code)
                        function_name.insert(0, function_name[0])
                        namespace.insert(0, "baseline")
                        logger.info(f"Auto-loaded baseline for {function_name[0]}")

            if _is_torch_op:
                # PyTorch operator: keep original logic
                if DISPATCH_TORCH_LIB:
                    checked_source = [self._check_code(s, fn_name, ns) for s, fn_name, ns in zip(source, function_name, namespace)]
            else:
                # Non-PyTorch operator: handle baseline and triton registration
                filtered_sources = []
                filtered_function_names = []
                filtered_namespaces = []

                for s, fn_name, ns in zip(source, function_name, namespace):
                    if ns == "baseline":
                        # Always register baseline to the "baseline" namespace
                        filtered_sources.append(s)
                        filtered_function_names.append(fn_name)
                        filtered_namespaces.append("baseline")

                        if not DISPATCH_TORCH_LIB:
                            # DISPATCH_TORCH_LIB=0: also register baseline to the "triton" namespace
                            filtered_sources.append(s)
                            filtered_function_names.append(fn_name)
                            filtered_namespaces.append("triton")
                    elif (ns == "triton" or ns == "") and DISPATCH_TORCH_LIB:
                        # DISPATCH_TORCH_LIB=1: register triton to the "triton" namespace
                        filtered_sources.append(s)
                        filtered_function_names.append(fn_name)
                        filtered_namespaces.append("triton")

                # Ensure _check_code() is always called
                checked_source = [self._check_code(s, fn_name, ns) for s, fn_name, ns in zip(filtered_sources, filtered_function_names, filtered_namespaces)]

            # TODO
            # should check the test_func
            if test_func is not None:
                for tf in test_func:
                    self._register_test_func_from_str(tf)
        except Exception as e:
            if result_queue is not None:
                result_queue.put(
                    VerifyResult(
                        op_name=function_name[0], 
                        success=False,
                        traceback=traceback.format_exc(), 
                        info={
                            "total": 0,
                            "failed": 0,
                            "success": 0,
                        }, 
                        code=source[-1],
                        test_func=test_func[-1] if test_func is not None else None,
                    )
                )
            return VerifyResult(
                op_name=function_name[0], 
                success=False,
                traceback=traceback.format_exc(), 
                code=source[-1],
                test_func=test_func[-1] if test_func is not None else None,
            )
        res = self.run_test(
            test_func_mark if test_func_mark else function_name[0], 
            self._running_config, 
        )
        res.code = source[-1]
        res.test_func = test_func[-1] if test_func else None
        if result_queue is not None:
            result_queue.put(res)
        return res

    @staticmethod
    def _register_test_func_from_str(code: str):
        assert "@label" in code, "No @label decorator found"
        if "@parametrize" not in code:
            console.rule(f"[bold yellow]No @parametrize decorator found")

        before = set(_label_registry.keys())
        # local_ns = {}
        try:
            exec(code, globals())
        except Exception as e:
            console.rule(f"[bold red]Verify failed: {e}")
            raise ValueError(f"Verify failed: {e}")
        after = set(_label_registry.keys())
        new_labels = after - before

        if not new_labels:
            console.rule("[bold red]No new test function found")
            raise ValueError("No new test function found")

        console.rule(f"[bold green]New test function found: {new_labels}")
        return

    def verify_test_func(
        self, 
        test_func_name: str, 
        test_func_code: str, 
        torch_kernel_name: str, 
        torch_kernel_code: str = None, 
        mark_suffix: str = None,
    ):
        # replace the "bench.triton." to "bench." and handle bench.triton.{torch_kernel_name} patterns
        import os
        os.environ["DISPATCH_TORCH_LIB"] = "0"
        os.environ["KERNELGENBENCH_UPCAST"] = "0"
        
        # First, do the simple replacement
        mocked_test_func_code = test_func_code.replace("kernelgenbench.triton.", "kernelgenbench.")
        
        # Then, handle more complex patterns like bench.triton.{torch_kernel_name}
        # This regex finds lines containing bench.triton.{torch_kernel_name} and replaces the entire line
        # pattern = rf'(\s*.*?)bench\.triton\.{re.escape(torch_kernel_name)}(.*?)(\n|$)'
        # replacement = rf'\1bench.{torch_kernel_name}\2\3'
        # mocked_test_func_code = re.sub(pattern, replacement, mocked_test_func_code, flags=re.MULTILINE)
        # mocked_test_func_code = test_func_code.replace(f"kernelgenbench.{torch_kernel_name}", f"torch.{torch_kernel_name}")
        
        results_with_mocked_test_func = self.only_verify(
            # name_source_map={
            #     torch_kernel_name: {
            #         "source": torch_kernel_code,
            #         "test_func": test_func_code,
            #     }
            # }
            name_source_map=[
                VerifyRequest(
                    source=[Source(
                        source=torch_kernel_code, 
                        function_name=torch_kernel_name,
                    )],
                    test_func=[mocked_test_func_code],
                    test_func_mark=torch_kernel_name if mark_suffix is None else torch_kernel_name + mark_suffix
                )
            ]
        )[1][0]
        results_with_mocked_test_func.test_func = test_func_code

        os.environ["DISPATCH_TORCH_LIB"] = "1"
        os.environ["KERNELGENBENCH_UPCAST"] = "1"
        return results_with_mocked_test_func

    def verify_triton_kernel(
        self, 
        triton_kernel_name: str,
        triton_kernel_code: str,
        torch_kernel_name: str, 
        torch_kernel_code: str, 
        test_func_name: str = None, 
        test_func_code: str = None, 
        benchmark_func_name: str = None, 
        benchmark_func_code: str = None,
        language: str = "zh_CN",        # en_US
    ):
        assert test_func_code is not None or benchmark_func_code is not None, "either test_func_code or benchmark_func_code should be provided"
        if test_func_code is not None:
            check_result = self.only_verify(
                # name_source_map={
                #     torch_kernel_name: {
                #         "source": torch_kernel_code,
                #         "test_func": test_func_code,
                #     }
                # }
                name_source_map=[
                    VerifyRequest(
                        source=[Source(
                            source=torch_kernel_code, 
                            function_name=torch_kernel_name, 
                            namespace=None, 
                        ), Source(
                            source=triton_kernel_code, 
                            function_name=triton_kernel_name,
                            namespace="triton",
                        )],
                        test_func=[test_func_code],
                        test_func_mark=torch_kernel_name
                    )
                ]
            )[1][0]

            if benchmark_func_code is None:
                return check_result

        # strict_check = self._running_config.strict_check

        # if check_result.success is not True:
        #     if not strict_check and "Tensor-likes are not close" in check_result.traceback:
        #         logger.warning("check failed, but ignore the error and continue to benchmark")
        #     else:
        #         return check_result
        
        benchmark_result = self.only_verify(
            name_source_map=[
                VerifyRequest(
                    source=[Source(
                        source=triton_kernel_code, 
                        function_name=triton_kernel_name,
                        namespace="triton",
                    ), Source(
                        source=torch_kernel_code, 
                        function_name=torch_kernel_name, 
                        namespace=None, 
                    )],
                    test_func=[benchmark_func_code],
                    test_func_mark=benchmark_func_name
                )
            ]
        )[1][0]
        if benchmark_result.speedup is None:
            return benchmark_result
        speedup_info = [su.model_dump() for su in benchmark_result.speedup]
        benchmark_result.info["html"] = generate_speedup_html(speedup_info, title="Performance Comparison Results", language=language)

        return benchmark_result