#!/usr/bin/env python3
"""
Generate test functions for K2 CUDA wrapper functions.

This script reads K2 CUDA wrapper functions from kernel_list_k2.py,
generates TestFuncGenerateArgs for each wrapper, and uses the test_func_generator
to generate corresponding test functions that compare PyTorch ground truth with Triton implementations.
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
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from flagbench.dataset.kernel_list_k2 import IMPL_INFO_K2
from generator.test_func_generator import TestFuncGenerator
from generator.sampler.generate_samples import (
    TestFuncGenerateArgs,
    GenerationConfig,
)


class K2TestFuncGenerator(TestFuncGenerator):
    """Custom TestFuncGenerator for K2 CUDA wrappers that preserves dynamic attributes."""
    
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
        # Check if this is a K2 CUDA wrapper test
        # Check both hasattr and __dict__ to support dynamically added attributes
        is_k2 = (hasattr(info, 'is_k2_cuda') and info.is_k2_cuda) or \
                (hasattr(info, '__dict__') and info.__dict__.get('is_k2_cuda', False))
        if is_k2:
            from rich.console import Console
            console = Console()
            console.print("Generating prompt for K2 CUDA wrapper test function...")
            return self.generate_prompt_for_k2_cuda(info)
        
        # Fall back to parent implementation
        return super().generate_prompt(info)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_ut_generate_args(kernel_name: str, impl_info: Dict[str, Any]) -> TestFuncGenerateArgs:
    """
    Create TestFuncGenerateArgs for a K2 CUDA wrapper function.
    
    Args:
        kernel_name: Name of the K2 kernel (e.g., 'AddEpsilonSelfLoops')
        impl_info: Implementation info from IMPL_INFO_K2
    
    Returns:
        TestFuncGenerateArgs instance (using dict to bypass Pydantic validation for extra fields)
    """
    # Create TestFuncGenerateArgs with only valid fields
    gen_arg = TestFuncGenerateArgs(
        kernel_name=kernel_name,
        operators=[],  # Empty for K2
        test_func_name=f"test_{kernel_name}",
    )
    
    # Add K2-specific attributes using __dict__ (Pydantic BaseModel allows this)
    # These will be accessible via hasattr() and getattr()
    gen_arg.__dict__['is_k2_cuda'] = True
    gen_arg.__dict__['impl_info'] = impl_info
    gen_arg.__dict__['problem_id'] = 0
    
    return gen_arg


def generate_samples(name: str, output_dir: Path, config: GenerationConfig, max_retries: int = 3, retry_delay: float = 1.0) -> None:
    """
    Generate test functions for K2 CUDA wrapper functions.
    
    Args:
        name: Name of the K2 kernel to generate, or "all" for all kernels
        output_dir: Directory to save generated results
        config: Generation configuration
    """
    # Get the list of K2 kernels to process
    if name.lower() == "all":
        kernels_to_process = list(IMPL_INFO_K2.keys())
        logger.info(f"Processing all {len(kernels_to_process)} K2 CUDA wrapper functions.")
    else:
        # Check if the specified name exists
        if name not in IMPL_INFO_K2:
            logger.error(f"Kernel '{name}' not found in IMPL_INFO_K2.")
            return
        kernels_to_process = [name]
        logger.info(f"Processing kernel: {name}")
    
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
    
    # Initialize the generator (use custom K2 generator to preserve dynamic attributes)
    generator = K2TestFuncGenerator(config)
    
    # Wrap the inference server with retry logic
    if hasattr(generator, 'inference_server'):
        original_server = generator.inference_server
        generator.inference_server = create_retry_inference_server(
            original_server,
            max_retries,
            retry_delay
        )
    
    # Process each kernel and collect generate args
    gen_args = []
    kernel_names = []
    for kernel_name in kernels_to_process:
        logger.info(f"Preparing: {kernel_name}")
        try:
            impl_info = IMPL_INFO_K2[kernel_name]
            # Create generate args for this kernel
            for sample_idx in range(config.num_samples):
                gen_arg = create_ut_generate_args(kernel_name, impl_info)
                gen_arg.sample_id = config.sample_id + sample_idx
                gen_args.append(gen_arg)
                kernel_names.append(kernel_name)
        except Exception as e:
            logger.error(f"✗ Error preparing {kernel_name}: {e}", exc_info=True)
    
    # Generate all test functions
    logger.info(f"Generating {len(gen_args)} test functions...")
    generated_codes = generator(gen_args)
    
    # Verify generated_codes is a list
    if not isinstance(generated_codes, list):
        logger.error(f"Expected list of generated codes, got {type(generated_codes)}")
        generated_codes = [generated_codes] if generated_codes else []
    
    # Process and save the generated codes
    results = []
    for idx, (generated_code, test_func_name, sample_id) in enumerate(generated_codes):
        kernel_name = kernel_names[idx] if idx < len(kernel_names) else test_func_name.replace("test_", "")
        logger.info(f"Processing result {idx + 1}/{len(kernel_names)}: {kernel_name}")
        
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
                    "kernel_name": kernel_name,
                    "test_func_name": test_func_name,
                    "file_path": str(test_path),
                    "success": True,
                    "code_length": len(generated_code),
                })
            except Exception as e:
                logger.error(f"✗ Error saving {kernel_name}: {e}")
                results.append({
                    "kernel_name": kernel_name,
                    "test_func_name": test_func_name,
                    "success": False,
                    "error": f"Save error: {str(e)}",
                })
        else:
            logger.warning(f"✗ Failed to generate test function for {kernel_name}")
            results.append({
                "kernel_name": kernel_name,
                "test_func_name": test_func_name,
                "success": False,
                "error": "Empty or invalid generation result",
            })
    
    # Calculate statistics
    total = len(kernel_names)
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
    logger.info(f"Total K2 Kernels: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {successful / total * 100:.2f}%" if total > 0 else "0%")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test functions for K2 CUDA wrapper functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for all K2 kernels
  python script/generate_ut_sample4k2.py --name all
  
  # Generate for a specific K2 kernel
  python script/generate_ut_sample4k2.py --name AddEpsilonSelfLoops
  
  # Use custom output directory
  python script/generate_ut_sample4k2.py --name all --output-dir ./my_output
  
  # Use different model
  python script/generate_ut_sample4k2.py --name all --server-type deepseek --model-name deepseek-coder
        """
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="all",
        help="Name of the K2 kernel to generate (default: all). Use 'all' to generate for all K2 kernels."
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "output",
        help="Output directory for generated samples (default: output)"
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
    run_name = f"ut_{args.model_name}_num_samples_{args.num_samples}_temp_{args.temperature}_max_tokens_{args.max_tokens}_{today()}"
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
    
    logger.info("Starting sample generation...")
    logger.info(f"Config: {config}")
    output_dir = args.output_dir / run_name
    # Generate samples
    generate_samples(args.name, output_dir, config, args.max_retries, args.retry_delay)
    
    logger.info("Sample generation completed!")


if __name__ == "__main__":
    main()
