#!/usr/bin/env python3
"""
Generate test functions for cuBLAS operations using CuPy baselines.

This script reads cuBLAS function schemas from cublas_ops.json,
generates TestFuncGenerateArgs for each function, and uses the test_func_generator
to generate corresponding test functions that compare Triton implementations with cuBLAS baselines (via CuPy).
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

def today() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from generator.test_func_generator import TestFuncGenerator
from generator.sampler.generate_samples import (
    TestFuncGenerateArgs,
    GenerationConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CuBLASTestFuncGenerator(TestFuncGenerator):
    """Custom TestFuncGenerator for cuBLAS operations that preserves dynamic attributes."""
    
    def _init_data(self, kwargs):
        # If kwargs is already a TestFuncGenerateArgs instance, preserve its __dict__ attributes
        if isinstance(kwargs, TestFuncGenerateArgs):
            # Preserve dynamically added attributes from __dict__ before calling parent
            extra_attrs = {k: v for k, v in kwargs.__dict__.items() 
                          if k not in ['kernel_name', 'operators', 'test_func_name', 
                                      'from_mcp', 'user_advice', 'check_result', 
                                      'old_code', 'sample_id']}
            # Call parent's _init_data which will return kwargs directly
            config = super()._init_data(kwargs)
            # Restore extra attributes after parent returns
            if extra_attrs:
                config.__dict__.update(extra_attrs)
            return config
        
        # Otherwise, call parent implementation
        return super()._init_data(kwargs)
    
    def generate_prompt(self, info: TestFuncGenerateArgs):
        # Check if this is a cuBLAS test
        # Check both hasattr and __dict__ to support dynamically added attributes
        use_cublas = (hasattr(info, 'use_cublas') and info.use_cublas) or \
                     (hasattr(info, '__dict__') and info.__dict__.get('use_cublas', False))
        
        if use_cublas:
            from rich.console import Console
            console = Console()
            console.print("[cyan]Generating prompt for cuBLAS test function...")
            # 调用已有的 generate_prompt_for_cublas 方法
            return self.generate_prompt_for_cublas(info)
        
        # Fall back to parent implementation
        return super().generate_prompt(info)


def load_cublas_schema(schema_path: Path) -> Dict[str, Any]:
    """Load cuBLAS schema from JSON file."""
    if not schema_path.exists():
        logger.error(f"cuBLAS schema file not found: {schema_path}")
        return None
    
    with open(schema_path, 'r') as f:
        return json.load(f)


def filter_valid_cublas_functions(functions: List[Dict]) -> List[Dict]:
    """
    Filter out non-BLAS functions (logger, handle management, etc.)
    and return only valid BLAS operations.
    """
    # 排除的操作类型（非 BLAS 核心操作）
    exclude_operations = {
        'loggerconfigure', 'etloggercallback', 'getloggercallback',
        'create', 'destroy', 'getversion', 'setstream', 'getstream',
        'getpointermode', 'setpointermode', 'setvector', 'getvector',
        'setmatrix', 'getmatrix', 'setvectorasync', 'getvectorasync',
        'setmatrixasync', 'getmatrixasync', 'setatomicsmode', 'getatomicsmode',
        'setmathmode', 'getmathmode'
    }
    
    valid_functions = []
    for func in functions:
        operation = func['operation'].lower()
        
        # 跳过排除的操作
        if operation in exclude_operations:
            continue
        
        # 跳过 _64 后缀的函数（64位整数变体）
        if func['name'].endswith('_64'):
            continue
        
        # 跳过 dtype 为 unknown 的函数（除非是确定的 BLAS 操作）
        if func['dtype'] == 'unknown' and operation not in {'iamax', 'iamin', 'asum', 'nrm2'}:
            continue
        
        valid_functions.append(func)
    
    return valid_functions


def get_baseline_operators() -> set:
    """
    Get the set of operator names from baseline directory.

    Returns:
        Set of operator names (e.g., {'sgemm', 'dgemm', 'caxpy', ...})
    """
    baseline_dir = PROJECT_ROOT / "src" / "flagbench" / "dataset" / "baseline" / "cupy"
    operators = set()
    
    if baseline_dir.exists():
        for f in baseline_dir.glob('*.py'):
            if f.name != '__init__.py':
                operators.add(f.stem)
    
    logger.info(f"Found {len(operators)} baseline operators")
    return operators


def filter_by_baseline(functions: List[Dict], baseline_ops: set) -> List[Dict]:
    """
    Filter functions to only include those that have baseline files.
    
    Args:
        functions: List of cuBLAS function dictionaries
        baseline_ops: Set of baseline operator names
    
    Returns:
        Filtered list of functions that have baseline files
    """
    filtered = []
    for func in functions:
        # 从函数名提取kernel名（如 "cublasSgemm_v2" -> "sgemm"）
        func_name = func['name']
        kernel_name = func_name.replace('cublas', '').replace('_v2', '').lower()
        
        if kernel_name in baseline_ops:
            filtered.append(func)
    
    logger.info(f"Filtered to {len(filtered)} functions with baseline files")
    return filtered


def create_ut_generate_args(cublas_func: Dict[str, Any]) -> TestFuncGenerateArgs:
    """
    Create TestFuncGenerateArgs for a cuBLAS function.
    
    Args:
        cublas_func: cuBLAS function schema from cublas_ops.json
    
    Returns:
        TestFuncGenerateArgs instance with cuBLAS-specific attributes
    """
    # 从完整函数名提取kernel名（如 "cublasSgemm_v2" -> "sgemm"）
    # 去掉 "cublas" 前缀和 "_v2" 后缀
    func_name = cublas_func['name']
    kernel_name = func_name.replace('cublas', '').replace('_v2', '').lower()
    
    # Create TestFuncGenerateArgs with only valid fields
    gen_arg = TestFuncGenerateArgs(
        kernel_name=kernel_name,
        operators={},  # Empty dict for cuBLAS
        test_func_name=f"test_{kernel_name}_cublas_baseline",
    )
    
    # Add cuBLAS-specific attributes using __dict__ (Pydantic BaseModel allows this)
    gen_arg.__dict__['use_cublas'] = True
    gen_arg.__dict__['cublas_schema'] = cublas_func
    gen_arg.__dict__['problem_id'] = 0
    gen_arg.__dict__['ops_namespace'] = 'aten'  # 默认使用 aten 命名空间
    
    return gen_arg


def generate_samples(
    name: str, 
    output_dir: Path, 
    config: GenerationConfig, 
    schema_path: Path,
    max_retries: int = 3, 
    retry_delay: float = 1.0
) -> None:
    """
    Generate test functions for cuBLAS operations.
    
    Args:
        name: Name of the cuBLAS operation to generate, "all" for all operations,
              or "baseline" for operators with baseline files
        output_dir: Directory to save generated results
        config: Generation configuration
        schema_path: Path to cublas_ops.json
        max_retries: Maximum retry attempts for failed requests
        retry_delay: Delay between retries (exponential backoff)
    """
    # Load cuBLAS schema
    schema = load_cublas_schema(schema_path)
    if schema is None:
        logger.error("Failed to load cuBLAS schema. Exiting.")
        return
    
    # Filter valid BLAS functions
    all_functions = filter_valid_cublas_functions(schema['functions'])
    logger.info(f"Loaded {len(all_functions)} valid cuBLAS functions (filtered from {schema['total_functions']} total)")
    
    # Get the list of functions to process
    if name.lower() == "all":
        functions_to_process = all_functions
        logger.info(f"Processing all {len(functions_to_process)} cuBLAS functions.")
    elif name.lower() == "baseline":
        # Only generate for operators with baseline files
        baseline_ops = get_baseline_operators()
        functions_to_process = filter_by_baseline(all_functions, baseline_ops)
        logger.info(f"Processing {len(functions_to_process)} cuBLAS functions (baseline operators only).")
    else:
        # Find functions matching the operation name
        matching_funcs = [f for f in all_functions if f['operation'].lower() == name.lower()]
        
        if not matching_funcs:
            logger.error(f"Operation '{name}' not found in cuBLAS schema.")
            logger.info(f"Available operations: {sorted(set(f['operation'] for f in all_functions))}")
            return
        
        functions_to_process = matching_funcs
        logger.info(f"Processing {len(functions_to_process)} function(s) for operation: {name}")
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_idx in range(config.num_samples):
        code_dir = output_dir / f"ut_{sample_idx}"
        code_dir.mkdir(exist_ok=True)
    
    # Create a retry wrapper for inference server
    def create_retry_inference_server(original_server, max_retries, retry_delay):
        """Wrap inference server with retry logic."""
        import time
        
        def retry_wrapper(prompt):
            for attempt in range(max_retries + 1):
                try:
                    return original_server(prompt)
                except Exception as e:
                    error_msg = str(e).lower()
                    is_retryable = any(keyword in error_msg for keyword in [
                        "connection", "timeout", "gateway", "503", "504", "502", "500"
                    ])
                    
                    if attempt < max_retries and is_retryable:
                        delay = retry_delay * (2 ** attempt)
                        logger.warning(
                            f"Retryable error (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                    else:
                        raise
            return None
        
        return retry_wrapper
    
    # Initialize the generator (use custom cuBLAS generator to preserve dynamic attributes)
    generator = CuBLASTestFuncGenerator(config)
    
    # Wrap the inference server with retry logic
    if hasattr(generator, 'inference_server'):
        original_server = generator.inference_server
        generator.inference_server = create_retry_inference_server(
            original_server,
            max_retries,
            retry_delay
        )
    
    # Process each function and collect generate args
    gen_args = []
    function_names = []
    
    for cublas_func in functions_to_process:
        func_name = cublas_func['name']
        operation = cublas_func['operation']
        dtype = cublas_func['dtype']
        
        logger.info(f"Preparing: {func_name} (operation={operation}, dtype={dtype})")
        
        try:
            # Create generate args for this function
            for sample_idx in range(config.num_samples):
                gen_arg = create_ut_generate_args(cublas_func)
                gen_arg.sample_id = config.sample_id + sample_idx
                gen_args.append(gen_arg)
                function_names.append(func_name)
        except Exception as e:
            logger.error(f"✗ Error preparing {func_name}: {e}", exc_info=True)
    
    # Generate all test functions
    logger.info(f"\n{'='*60}")
    logger.info(f"Generating {len(gen_args)} test functions...")
    logger.info(f"{'='*60}\n")
    
    generated_codes = generator(gen_args)
    
    # Verify generated_codes is a list
    if not isinstance(generated_codes, list):
        logger.error(f"Expected list of generated codes, got {type(generated_codes)}")
        generated_codes = [generated_codes] if generated_codes else []
    
    # Process and save the generated codes
    results = []
    for idx, (generated_code, test_func_name, sample_id) in enumerate(generated_codes):
        func_name = function_names[idx] if idx < len(function_names) else test_func_name.replace("test_", "")
        logger.info(f"Processing result {idx + 1}/{len(function_names)}: {func_name}")
        
        # Check if generation was successful
        if generated_code and isinstance(generated_code, str) and len(generated_code.strip()) > 0:
            try:
                # Save to file
                test_filename = f"{test_func_name}.py"
                test_path = output_dir / f"ut_{sample_id}" / test_filename
                
                with open(test_path, "w") as f:
                    f.write(generated_code)
                
                logger.info(f"✓ Generated test function saved to: {test_path}")
                
                # Record successful result
                results.append({
                    "cublas_function": func_name,
                    "test_func_name": test_func_name,
                    "file_path": str(test_path),
                    "success": True,
                    "code_length": len(generated_code),
                })
            except Exception as e:
                logger.error(f"✗ Error saving {func_name}: {e}")
                results.append({
                    "cublas_function": func_name,
                    "test_func_name": test_func_name,
                    "success": False,
                    "error": f"Save error: {str(e)}",
                })
        else:
            logger.warning(f"✗ Failed to generate test function for {func_name}")
            results.append({
                "cublas_function": func_name,
                "test_func_name": test_func_name,
                "success": False,
                "error": "Empty or invalid generation result",
            })
    
    # Calculate statistics
    total = len(function_names)
    successful = sum(1 for r in results if r["success"])
    failed = total - successful
    
    # Save detailed summary
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{successful / total * 100:.2f}%" if total > 0 else "0%",
            "results": results,
        }, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Generation Summary:")
    logger.info(f"Total cuBLAS Functions: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {successful / total * 100:.2f}%" if total > 0 else "0%")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test functions for cuBLAS operations using CuPy baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for all cuBLAS operations
  python script/cublas_cupy/generate_ut_cublas.py --name all
  
  # Generate for a specific operation (e.g., gemm)
  python script/cublas_cupy/generate_ut_cublas.py --name gemm
  
  # Use custom output directory
  python script/cublas_cupy/generate_ut_cublas.py --name all --output-dir ./my_output
  
  # Use different model
  python script/cublas_cupy/generate_ut_cublas.py --name all --server-type deepseek --model-name deepseek-coder
        """
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="all",
        help="Name of the cuBLAS operation to generate (default: all). Use 'all' to generate for all operations."
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output_ut_cublas",
        help="Output directory for generated samples (default: output_ut_cublas)"
    )
    
    parser.add_argument(
        "--schema-path",
        type=Path,
        default=SCRIPT_DIR / "cublas_ops.json",
        help="Path to cublas_ops.json schema file (default: script/cublas_cupy/cublas_ops.json)"
    )
    
    parser.add_argument(
        "--server-type",
        type=str,
        default="panda",
        choices=["qwen", "deepseek", "openai", "anthropic", "google", "together", "sglang", "vllm", "panda"],
        help="LLM server type to use (default: panda)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="gpt-5",
        help="Model name to use (default: gpt-5)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for generation (default: 0.0)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens to generate (default: 16384)"
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate (default: 1)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10, recommended: 5-20 for API stability)"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts for failed requests (default: 3)"
    )
    
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Initial delay between retries in seconds (default: 1.0, uses exponential backoff)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set verbose logging if requested
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    run_name = f"ut_cublas_{args.model_name}_num_samples_{args.num_samples}_temp_{args.temperature}_max_tokens_{args.max_tokens}_{today()}"
    
    # Create generation config
    config = GenerationConfig(
        run_name=run_name,
        server_type=args.server_type,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        num_samples=args.num_samples,
        verbose=args.verbose,
        run_dir=str(args.output_dir),
        log_prompt=True,
    )
    
    logger.info("Starting cuBLAS test function generation...")
    logger.info(f"Config: {config}")
    
    output_dir = args.output_dir / run_name
    
    # Generate samples
    generate_samples(
        args.name, 
        output_dir, 
        config, 
        args.schema_path,
        args.max_retries, 
        args.retry_delay
    )
    
    logger.info("cuBLAS test function generation completed!")


if __name__ == "__main__":
    main()
