#!/usr/bin/env python3
"""
Generate benchmark functions for cuBLAS operations using CuPy baselines.

This script reads cuBLAS function schemas from cublas_ops.json,
generates BenchmarkFuncGenerateArgs for each function, and uses the benchmark_func_generator
to generate corresponding benchmark functions that compare Triton implementations with cuBLAS baselines (via CuPy).
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

from generator.benchmark_func_generator import BenchmarkFuncGenerator
from generator.sampler.generate_samples import (
    BenchmarkFuncGenerateArgs,
    GenerationConfig,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CuBLASBenchmarkFuncGenerator(BenchmarkFuncGenerator):
    """Custom BenchmarkFuncGenerator for cuBLAS operations that preserves dynamic attributes."""
    
    def _init_data(self, kwargs):
        # If kwargs is already a BenchmarkFuncGenerateArgs instance, preserve its __dict__ attributes
        if isinstance(kwargs, BenchmarkFuncGenerateArgs):
            # Preserve dynamically added attributes from __dict__ before calling parent
            extra_attrs = {k: v for k, v in kwargs.__dict__.items() 
                          if k not in ['kernel_name', 'test_perf_func_name', 'ops_namespace',
                                      'test_func_code', 'user_advice', 'check_result', 
                                      'old_code', 'sample_id']}
            # Call parent's _init_data which will return kwargs directly
            config = super()._init_data(kwargs)
            # Restore extra attributes after parent returns
            if extra_attrs:
                config.__dict__.update(extra_attrs)
            return config
        
        # Otherwise, call parent implementation
        return super()._init_data(kwargs)


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


def create_benchmark_generate_args(cublas_func: Dict[str, Any], ut_file_path: str = None) -> BenchmarkFuncGenerateArgs:
    """
    Create BenchmarkFuncGenerateArgs for a cuBLAS function.
    
    Args:
        cublas_func: cuBLAS function schema from cublas_ops.json
        ut_file_path: Path to corresponding UT file (optional, for reading test_func_code)
    
    Returns:
        BenchmarkFuncGenerateArgs instance with cuBLAS-specific attributes
    """
    # 从完整函数名提取kernel名（如 "cublasSgemm_v2" -> "sgemm"）
    # 去掉 "cublas" 前缀和 "_v2" 后缀
    func_name = cublas_func['name']
    kernel_name = func_name.replace('cublas', '').replace('_v2', '').lower()
    
    # Read test function code if UT file path is provided
    test_func_code = None
    if ut_file_path and os.path.exists(ut_file_path):
        try:
            with open(ut_file_path, 'r') as f:
                test_func_code = f.read()
        except Exception as e:
            logger.warning(f"Failed to read UT file {ut_file_path}: {e}")
    
    # Create BenchmarkFuncGenerateArgs with only valid fields
    gen_arg = BenchmarkFuncGenerateArgs(
        kernel_name=kernel_name,
        test_perf_func_name=f"test_perf_{kernel_name}_cublas_baseline",
        ops_namespace="cublas",  # Use cublas namespace
        test_func_code=test_func_code,
    )
    
    # Add cuBLAS-specific attributes using __dict__ (Pydantic BaseModel allows this)
    gen_arg.__dict__['cublas_schema'] = cublas_func
    
    return gen_arg


def main():
    parser = argparse.ArgumentParser(
        description="Generate benchmark functions for cuBLAS operations using CuPy baselines"
    )
    
    # Operation selection
    parser.add_argument(
        "--name",
        type=str,
        default="gemm",
        help="Operation name to generate benchmarks for (e.g., 'gemm', 'axpy', 'all'). Default: 'gemm'"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of benchmark samples to generate per operation. Default: 1"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of parallel workers for generation. Default: 10"
    )
    
    # Model parameters
    parser.add_argument(
        "--model-name",
        type=str,
        default="deepseek-v3-0324",
        help="LLM model name. Default: deepseek-v3-0324"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Maximum tokens for LLM response. Default: 16384"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM temperature. Default: 0.0"
    )
    
    # Output configuration
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Output directory for generated benchmarks. Default: <project_root>/output_benchmark_cublas"
    )
    
    parser.add_argument(
        "--ut-dir",
        type=str,
        default=None,
        help="Directory containing UT files (optional, for reading test_func_code)"
    )
    
    # Other options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set up paths
    schema_path = SCRIPT_DIR / "cublas_ops.json"
    
    if args.run_dir is None:
        run_dir = PROJECT_ROOT / "output_benchmark_cublas"
    else:
        run_dir = Path(args.run_dir)
    
    # Create run name
    run_name = f"benchmark_cublas_{args.model_name}_num_samples_{args.num_samples}_temp_{args.temperature}_max_tokens_{args.max_tokens}_{today()}"
    
    # Create GenerationConfig
    config = GenerationConfig(
        run_name=run_name,
        sample_id=0,
        num_samples=args.num_samples,
        test_type="accuracy",  # Not used for benchmarks, but required by GenerationConfig
        num_workers=args.num_workers,
        server_type="panda",
        model_name=args.model_name,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        run_dir=str(run_dir),
        save_log=True,
        verbose=args.verbose,
        store_type="local",
        log_prompt=True,
        backend="triton",
    )
    
    logger.info("Starting cuBLAS benchmark function generation...")
    logger.info(f"Config: {config}")
    
    # Load cuBLAS schema
    schema = load_cublas_schema(schema_path)
    if schema is None:
        logger.error("Failed to load cuBLAS schema")
        return 1
    
    functions = schema.get('functions', [])
    
    # Filter valid functions
    valid_functions = filter_valid_cublas_functions(functions)
    logger.info(f"Loaded {len(valid_functions)} valid cuBLAS functions (filtered from {len(functions)} total)")
    
    # Filter by operation name
    if args.name.lower() == "all":
        target_functions = valid_functions
    else:
        target_operation = args.name.lower()
        target_functions = [f for f in valid_functions if f['operation'].lower() == target_operation]
    
    if not target_functions:
        logger.error(f"No cuBLAS functions found for operation: {args.name}")
        return 1
    
    logger.info(f"Processing {len(target_functions)} function(s) for operation: {args.name}")
    
    # Create BenchmarkFuncGenerateArgs for each function
    benchmark_args_list = []
    for cublas_func in target_functions:
        func_name = cublas_func['name']
        kernel_name = func_name.replace('cublas', '').replace('_v2', '').lower()
        
        logger.info(f"Preparing: {func_name} (operation={cublas_func['operation']}, dtype={cublas_func['dtype']})")
        
        # Try to find corresponding UT file if ut_dir is provided
        ut_file_path = None
        if args.ut_dir:
            ut_file_path = Path(args.ut_dir) / f"test_{kernel_name}_cublas_baseline.py"
            if not ut_file_path.exists():
                logger.warning(f"UT file not found: {ut_file_path}")
                ut_file_path = None
        
        gen_arg = create_benchmark_generate_args(cublas_func, ut_file_path=str(ut_file_path) if ut_file_path else None)
        benchmark_args_list.append(gen_arg)
    
    # Generate benchmark functions
    logger.info("=" * 70)
    logger.info(f"Generating {len(benchmark_args_list)} benchmark functions...")
    logger.info("=" * 70)
    
    generator = CuBLASBenchmarkFuncGenerator(config)
    results = generator.generate(benchmark_args_list)
    
    # Process and save results
    output_dir = run_dir / run_name / "benchmark_0"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    failed_count = 0
    
    for i, (gen_arg, result) in enumerate(zip(benchmark_args_list, results)):
        cublas_func = gen_arg.__dict__.get('cublas_schema', {})
        func_name = cublas_func.get('name', f'unknown_{i}')
        kernel_name = gen_arg.kernel_name
        
        logger.info(f"Processing result {i+1}/{len(results)}: {func_name}")
        
        if result and len(result.strip()) > 0:
            output_file = output_dir / f"benchmark_{kernel_name}_cublas_baseline.py"
            
            # Add standard imports at the beginning
            imports = """import flagbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton
import cupy as cp
from torch.utils.dlpack import to_dlpack, from_dlpack

"""
            full_code = imports + result
            
            with open(output_file, 'w') as f:
                f.write(full_code)
            
            logger.info(f"✓ Generated benchmark function saved to: {output_file}")
            success_count += 1
        else:
            logger.error(f"✗ Failed to generate benchmark for {func_name}")
            failed_count += 1
    
    # Save generation summary
    summary = {
        "total_functions": len(target_functions),
        "successful": success_count,
        "failed": failed_count,
        "operation": args.name,
        "model": args.model_name,
        "timestamp": today(),
        "output_dir": str(output_dir),
    }
    
    summary_file = run_dir / run_name / "generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 70)
    logger.info("Generation Summary:")
    logger.info(f"Total cuBLAS Functions: {len(target_functions)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Success Rate: {100 * success_count / len(target_functions) if target_functions else 0:.2f}%")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("=" * 70)
    
    logger.info("cuBLAS benchmark function generation completed!")
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
