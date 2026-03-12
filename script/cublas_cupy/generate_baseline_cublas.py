#!/usr/bin/env python3
"""
Generate baseline functions for cuBLAS operations using CuPy.

This script uses LLM to generate baseline wrapper functions that call cuBLAS via CuPy.
The LLM intelligently chooses between direct cupy.cublas calls and CuPy array operations.
These baselines are registered to flagbench.baseline namespace.
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

from generator.baseline_func_generator import BaselineFuncGenerator
from generator.sampler.generate_samples import GenerationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def generate_baselines(
    name: str, 
    output_dir: Path, 
    config: GenerationConfig, 
    schema_path: Path,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> None:
    """
    Generate baseline functions for cuBLAS operations using LLM.
    
    Args:
        name: Name of the cuBLAS operation to generate, or "all" for all operations
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
    code_dir = output_dir / "baseline_0"
    code_dir.mkdir(exist_ok=True)
    
    # Create retry wrapper for inference server
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
    
    # Initialize the generator
    generator = BaselineFuncGenerator(config)
    
    # Wrap the inference server with retry logic
    if hasattr(generator, 'inference_server'):
        original_server = generator.inference_server
        generator.inference_server = create_retry_inference_server(
            original_server,
            max_retries,
            retry_delay
        )
    
    # Generate baseline functions
    logger.info("=" * 70)
    logger.info(f"Generating {len(functions_to_process)} baseline functions using LLM...")
    logger.info("=" * 70)
    
    # Call the generator with list of cuBLAS functions
    generated_codes = generator(functions_to_process)
    
    # Process and save the generated codes
    success_count = 0
    failed_count = 0
    
    for idx, (cublas_func, generated_code) in enumerate(zip(functions_to_process, generated_codes)):
        func_name = cublas_func['name']
        kernel_name = func_name.replace('cublas', '').replace('_v2', '').lower()
        
        logger.info(f"Processing {idx + 1}/{len(functions_to_process)}: {func_name} -> {kernel_name}.py")
        
        # Check if generation was successful
        if generated_code and isinstance(generated_code, str) and len(generated_code.strip()) > 0:
            try:
                # Save to file
                output_file = code_dir / f"{kernel_name}.py"
                with open(output_file, 'w') as f:
                    f.write(generated_code)
                
                logger.info(f"✓ Baseline saved to: {output_file}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"✗ Error saving {kernel_name}: {e}")
                failed_count += 1
        else:
            logger.warning(f"✗ Failed to generate baseline for {func_name}")
            failed_count += 1
    
    # Save generation summary
    summary = {
        "total_functions": len(functions_to_process),
        "successful": success_count,
        "failed": failed_count,
        "operation": name,
        "timestamp": today(),
        "output_dir": str(code_dir),
    }
    
    summary_file = output_dir / "generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("=" * 70)
    logger.info("Generation Summary:")
    logger.info(f"Total cuBLAS Functions: {len(functions_to_process)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Success Rate: {100 * success_count / len(functions_to_process) if functions_to_process else 0:.2f}%")
    logger.info(f"Summary saved to: {summary_file}")
    logger.info("=" * 70)
    
    logger.info("cuBLAS baseline function generation completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Generate baseline functions for cuBLAS operations using CuPy and LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for a specific operation (e.g., gemm)
  python script/cublas_cupy/generate_baseline_cublas.py --name gemm
  
  # Generate for all cuBLAS operations
  python script/cublas_cupy/generate_baseline_cublas.py --name all
  
  # Use custom output directory
  python script/cublas_cupy/generate_baseline_cublas.py --name gemm --run-dir ./my_output
  
  # Use different model
  python script/cublas_cupy/generate_baseline_cublas.py --name all --server-type deepseek --model-name deepseek-coder
        """
    )
    
    # Operation selection
    parser.add_argument(
        "--name",
        type=str,
        default="gemm",
        help="Operation name to generate baselines for (e.g., 'gemm', 'axpy', 'all'). Default: 'gemm'"
    )
    
    # Output configuration
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Output directory for generated baselines. Default: <project_root>/output_baseline_cublas"
    )
    
    # LLM configuration
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
        default="deepseek-v3-0324",
        help="Model name to use (default: deepseek-v3-0324)"
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
        default=8192,
        help="Maximum tokens to generate (default: 8192)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)"
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
        help="Initial delay between retries in seconds (default: 1.0)"
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
        run_dir = PROJECT_ROOT / "output_baseline_cublas"
    else:
        run_dir = Path(args.run_dir)
    
    # Create run name
    run_name = f"baseline_cublas_{args.model_name}_temp_{args.temperature}_{today()}"
    
    logger.info("Starting cuBLAS baseline function generation using LLM...")
    logger.info(f"Output directory: {run_dir / run_name}")
    
    # Create generation config
    config = GenerationConfig(
        run_name=run_name,
        server_type=args.server_type,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        num_samples=1,  # Always 1 for baselines
        verbose=args.verbose,
        run_dir=str(run_dir),
        log_prompt=args.verbose,
    )
    
    logger.info(f"Config: {config}")
    
    output_dir = run_dir / run_name
    
    # Generate baselines
    generate_baselines(
        args.name, 
        output_dir, 
        config, 
        schema_path,
        args.max_retries,
        args.retry_delay
    )


if __name__ == "__main__":
    sys.exit(main())
