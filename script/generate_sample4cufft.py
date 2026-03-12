#!/usr/bin/env python3
"""
Generate sample script for creating Triton kernels from PyTorch APIs.

This script reads PyTorch APIs from the kernel list in src/flagbench/dataset,
generates TritonKernelGenerateArgs for each API, and uses the triton_kernel_generator
to generate corresponding Triton implementations.
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

from flagbench.dataset.kernel_list_cufft import IMPL_INFO_CUFFT
from generator.triton_kernel_generator4cufft import TritonKernelGenerator4CuFFT
from generator.sampler.generate_samples import (
    TritonKernelGenerateArgs,
    GenerationConfig,
    InputArg,
    OutputArg,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_cufft_reference_code(kernel_name: str, impl_info: Dict[str, Any]) -> str:
    """
    Create comprehensive reference code for cuFFT function.
    
    Args:
        kernel_name: Name of the cuFFT function (e.g., 'fft_2d_c2c')
        impl_info: Implementation info from IMPL_INFO_CUFFT
    
    Returns:
        Formatted reference code string for LLM prompt
    """
    sections = []
    sections.append(f"# cuFFT Function: {kernel_name}")
    sections.append(f"# Description: {impl_info.get('description', 'cuFFT operation')}")
    sections.append("")
    
    # Add cuFFT API usage example
    sections.append("# cuFFT API Usage (internal implementation):")
    sections.append("# This function internally combines:")
    
    # Determine which cuFFT APIs are used based on function name
    if "1d" in kernel_name:
        plan_func = "cufftPlan1d"
    elif "2d" in kernel_name:
        plan_func = "cufftPlan2d"
    elif "3d" in kernel_name:
        plan_func = "cufftPlan3d"
    else:
        plan_func = "cufftPlan*"
    
    # Determine exec function based on type
    if "c2c" in kernel_name or "z2z" in kernel_name:
        exec_func = "cufftExecC2C" if "c2c" in kernel_name else "cufftExecZ2Z"
        sections.append(f"#   1. {plan_func}(&plan, nx, ny, CUFFT_C2C)  // Create plan")
        sections.append(f"#   2. {exec_func}(plan, input, output, direction)  // Execute FFT")
    elif "r2c" in kernel_name or "d2z" in kernel_name:
        exec_func = "cufftExecR2C" if "r2c" in kernel_name else "cufftExecD2Z"
        sections.append(f"#   1. {plan_func}(&plan, nx, ny, CUFFT_R2C)  // Create plan")
        sections.append(f"#   2. {exec_func}(plan, input, output)  // Execute FFT")
    elif "c2r" in kernel_name or "z2d" in kernel_name:
        exec_func = "cufftExecC2R" if "c2r" in kernel_name else "cufftExecZ2D"
        sections.append(f"#   1. {plan_func}(&plan, nx, ny, CUFFT_C2R)  // Create plan")
        sections.append(f"#   2. {exec_func}(plan, input, output)  // Execute FFT")
    
    sections.append("#   3. cufftDestroy(plan)  // Destroy plan")
    sections.append("")
    
    # Add input/output args info
    if "input_args" in impl_info:
        sections.append("# Input Arguments:")
        for arg in impl_info["input_args"]:
            sections.append(f"#   - {arg.get('name')}: {arg.get('type')}")
        sections.append("")
    
    if "output_args" in impl_info:
        sections.append("# Output Arguments:")
        for arg in impl_info["output_args"]:
            sections.append(f"#   - {arg.get('name')}: {arg.get('type')}")
        sections.append("")
    
    return "\n".join(sections).strip()


def create_triton_generate_args(kernel_name: str, impl_info: Dict[str, Any]) -> TritonKernelGenerateArgs:
    """
    Create TritonKernelGenerateArgs for a cuFFT function.
    
    Args:
        kernel_name: Name of the cuFFT function (e.g., 'fft_2d_c2c')
        impl_info: Implementation info from IMPL_INFO_CUFFT
    
    Returns:
        TritonKernelGenerateArgs instance
    """
    # Create comprehensive reference code
    torch_kernel_code = create_cufft_reference_code(kernel_name, impl_info)
    
    # Extract input/output args
    input_args = []
    if "input_args" in impl_info and impl_info["input_args"]:
        for arg in impl_info["input_args"]:
            input_args.append(InputArg(
                arg_name=arg.get("name", "arg"),
                arg_type=arg.get("type", "Any"),
                arg_desc=arg.get("desc", "")
            ))
    else:
        input_args = [InputArg(arg_name="*args", arg_type="Any", arg_desc="Input arguments")]
    
    output_args = []
    if "output_args" in impl_info and impl_info["output_args"]:
        for arg in impl_info["output_args"]:
            output_args.append(OutputArg(
                arg_type=arg.get("type", "void"),
                arg_desc=arg.get("desc", "")
            ))
    else:
        output_args = [OutputArg(arg_type="void", arg_desc="In-place operation")]
    
    func_desc = impl_info.get("description", f"cuFFT function: {kernel_name}")
    
    # op_name is a property that returns triton_kernel_name, so we just set triton_kernel_name
    return TritonKernelGenerateArgs(
        triton_kernel_name=kernel_name,
        func_desc=func_desc,
        torch_kernel_code=torch_kernel_code,
        input_args=input_args,
        output_args=output_args,
        impl_info=None,  # cuFFT doesn't use multi-operator pattern
        from_mcp=False,
    )


def generate_samples(name: str, output_dir: Path, config: GenerationConfig) -> None:
    """
    Generate Triton kernel samples for the specified APIs.
    
    Args:
        name: Name of the API to generate, or "all" for all APIs
        output_dir: Directory to save generated results
        config: Generation configuration
    """
    # Get the list of cuFFT functions to process
    if name.lower() == "all":
        kernels_to_process = IMPL_INFO_CUFFT
        logger.info(f"Processing all {len(kernels_to_process)} cuFFT functions")
    else:
        # Check if the specified name exists
        if name not in IMPL_INFO_CUFFT:
            logger.error(f"cuFFT function '{name}' not found in IMPL_INFO_CUFFT")
            logger.info(f"Available cuFFT functions: {list(IMPL_INFO_CUFFT.keys())}")
            return
        kernels_to_process = {name: IMPL_INFO_CUFFT[name]}
        logger.info(f"Processing single cuFFT function: {name}")
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_idx in range(config.num_samples):
        code_dir = output_dir / f"code_{sample_idx}"
        code_dir.mkdir(exist_ok=True)
    
    # Initialize the generator
    generator = TritonKernelGenerator4CuFFT(config)
    
    # Process each cuFFT function and collect generate args
    gen_args = []
    kernel_names = []
    
    for kernel_name, impl_info in kernels_to_process.items():
        logger.info(f"Preparing: {kernel_name}")
        
        try:
            # Create generate args for this K1 wrapper
            for sample_idx in range(config.num_samples):
                gen_arg = create_triton_generate_args(kernel_name, impl_info)
                gen_arg.sample_id = config.sample_id + sample_idx
                gen_args.append(gen_arg)
                kernel_names.append(kernel_name)
        except Exception as e:
            logger.error(f"✗ Error preparing {kernel_name}: {e}", exc_info=True)
    
    # Generate all Triton kernels
    logger.info(f"Generating {len(gen_args)} Triton kernels...")
    generated_codes = generator(gen_args)
    
    # Verify generated_codes is a list
    if not isinstance(generated_codes, list):
        logger.error(f"Expected list of generated codes, got {type(generated_codes)}")
        generated_codes = [generated_codes] if generated_codes else []
    
    # Process and save the generated codes
    results = []
    for idx, (generated_code, name, sample_id) in enumerate(generated_codes):
        kernel_name = kernel_names[idx] if idx < len(kernel_names) else name
        logger.info(f"Processing result {idx + 1}/{len(kernel_names)}: {name}")
        
        # Check if generation was successful
        if generated_code and isinstance(generated_code, str) and len(generated_code.strip()) > 0:
            try:
                # Save to file
                kernel_filename = f"{name}.py"
                kernel_path = output_dir / f"code_{sample_id}" / kernel_filename
                
                with open(kernel_path, "w") as f:
                    f.write(generated_code)
                
                logger.info(f"✓ Generated kernel saved to: {kernel_path}")
                
                # Record successful result
                results.append({
                    "kernel_name": kernel_name,
                    "triton_name": name,
                    "file_path": str(kernel_path),
                    "success": True,
                    "code_length": len(generated_code),
                })
            except Exception as e:
                logger.error(f"✗ Error saving {kernel_name}: {e}")
                results.append({
                    "kernel_name": kernel_name,
                    "triton_name": name,
                    "success": False,
                    "error": f"Save error: {str(e)}",
                })
        else:
            logger.warning(f"✗ Failed to generate code for {kernel_name}")
            results.append({
                "kernel_name": kernel_name,
                "triton_name": name,
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
    logger.info(f"Total cuFFT Functions: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {successful / total * 100:.2f}%" if total > 0 else "0%")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Triton kernels from cuFFT functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for all cuFFT functions
  python script/generate_sample4cufft.py --name all
  
  # Generate for a specific cuFFT function
  python script/generate_sample4cufft.py --name fft_2d_c2c
  
  # Use custom output directory
  python script/generate_sample4cufft.py --name all --output-dir ./my_output
  
  # Use different model
  python script/generate_sample4cufft.py --name all --server-type deepseek --model-name deepseek-coder
        """
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="all",
        help="Name of the cuFFT function to generate (default: all). Use 'all' to generate for all cuFFT functions. Use function name (e.g., 'fft_2d_c2c')."
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
        default=150,
        help="Number of parallel workers (default: 1)"
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
    run_name = f"{args.model_name}_num_samples_{args.num_samples}_temp_{args.temperature}_max_tokens_{args.max_tokens}_{today()}"
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
    generate_samples(args.name, output_dir, config)
    
    logger.info("Sample generation completed!")


if __name__ == "__main__":
    main()
