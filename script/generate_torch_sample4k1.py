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

from flagbench.dataset.kernel_list_k1 import IMPL_INFO_K1
from generator.torch_kernel_generator import TorchKernelGenerator
from generator.sampler.generate_samples import (
    TorchKernelGenerateArgs,
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


def create_k1_reference_code(kernel_name: str, impl_info: Dict[str, Any]) -> str:
    """
    Create comprehensive reference code for K1 CUDA kernel.
    
    Args:
        kernel_name: Name of the K1 kernel (e.g., 'copy_low_upp')
        impl_info: Implementation info from IMPL_INFO_K1
    
    Returns:
        Formatted reference code string for LLM prompt
    """
    sections = []
    sections.append(f"# K1 CUDA Wrapper: cuda_{kernel_name}")
    sections.append(f"# Description: {impl_info.get('description', 'K1 CUDA operation')}")
    sections.append("")
    
    # Add algorithm description if available
    if "algorithm" in impl_info:
        sections.append(f"# Algorithm: {impl_info['algorithm']}")
        sections.append("")
    
    # Add CUDA kernel code
    if "cuda_kernel_code" in impl_info and impl_info["cuda_kernel_code"]:
        sections.append("# CUDA Kernel Implementation:")
        sections.append("```cuda")
        sections.append(impl_info["cuda_kernel_code"].strip())
        sections.append("```")
        sections.append("")
    
    # Add wrapper code
    if "wrapper_code" in impl_info and impl_info["wrapper_code"]:
        sections.append("# Wrapper Code:")
        sections.append("```cpp")
        sections.append(impl_info["wrapper_code"].strip())
        sections.append("```")
        sections.append("")
    
    # Add usage example if available
    if "usage_example" in impl_info and impl_info["usage_example"]:
        sections.append("# Usage Example:")
        sections.append("```cpp")
        sections.append(impl_info["usage_example"].strip())
        sections.append("```")
        sections.append("")
    
    # Add hints if available
    if "hints" in impl_info and impl_info["hints"]:
        sections.append(f"# Hints: {impl_info['hints']}")
        sections.append("")
    
    return "\n".join(sections).strip()


def create_torch_generate_args(kernel_name: str, impl_info: Dict[str, Any]) -> TorchKernelGenerateArgs:
    """
    Create TorchKernelGenerateArgs for a K1 CUDA kernel.
    
    Args:
        kernel_name: Name of the K1 kernel (e.g., 'copy_low_upp')
        impl_info: Implementation info from IMPL_INFO_K1
    
    Returns:
        TorchKernelGenerateArgs instance
    """
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
    
    func_desc = impl_info.get("description", f"K1 CUDA wrapper: {kernel_name}")
    
    return TorchKernelGenerateArgs(
        torch_kernel_name=kernel_name,
        func_desc=func_desc,
        input_args=input_args,
        output_args=output_args,
        func_type="k1_cuda",  # Mark as K1 CUDA wrapper for prompt detection
    )


def generate_samples(name: str, output_dir: Path, config: GenerationConfig, max_retries: int = 3, retry_delay: float = 1.0) -> None:
    """
    Generate PyTorch kernel samples for the specified K1 CUDA wrappers.
    
    Args:
        name: Name of the K1 wrapper to generate, or "all" for all wrappers
        output_dir: Directory to save generated results
        config: Generation configuration
    """
    # Get the list of K1 kernels to process
    if name.lower() == "all":
        kernels_to_process = IMPL_INFO_K1
        logger.info(f"Processing all {len(kernels_to_process)} K1 CUDA wrappers")
    else:
        # Check if the specified name exists
        if name not in IMPL_INFO_K1:
            logger.error(f"K1 wrapper '{name}' not found in IMPL_INFO_K1")
            logger.info(f"Available K1 wrappers: {list(IMPL_INFO_K1.keys())}")
            return
        kernels_to_process = {name: IMPL_INFO_K1[name]}
        logger.info(f"Processing single K1 wrapper: {name}")
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_idx in range(config.num_samples):
        code_dir = output_dir / f"groundtruth_{sample_idx}"
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
    
    # Initialize the generator
    generator = TorchKernelGenerator(config)
    
    # Wrap the inference server with retry logic
    if hasattr(generator, 'inference_server'):
        original_server = generator.inference_server
        generator.inference_server = create_retry_inference_server(
            original_server,
            max_retries,
            retry_delay
        )
    
    # Process each K1 wrapper and collect generate args
    gen_args = []
    kernel_names = []
    
    for kernel_name, impl_info in kernels_to_process.items():
        logger.info(f"Preparing: cuda_{kernel_name}")
        
        try:
            # Create generate args for this K1 wrapper
            for sample_idx in range(config.num_samples):
                gen_arg = create_torch_generate_args(kernel_name, impl_info)
                gen_arg.sample_id = config.sample_id + sample_idx
                gen_args.append(gen_arg)
                kernel_names.append(kernel_name)
        except Exception as e:
            logger.error(f"✗ Error preparing {kernel_name}: {e}", exc_info=True)
    
    # Generate all PyTorch kernels
    logger.info(f"Generating {len(gen_args)} PyTorch kernels...")
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
                kernel_filename = f"torch_{name}.py"
                kernel_path = output_dir / f"groundtruth_{sample_id}" / kernel_filename
                
                with open(kernel_path, "w") as f:
                    f.write(generated_code)
                
                logger.info(f"✓ Generated kernel saved to: {kernel_path}")
                
                # Record successful result
                results.append({
                    "kernel_name": kernel_name,
                    "torch_name": name,
                    "file_path": str(kernel_path),
                    "success": True,
                    "code_length": len(generated_code),
                })
            except Exception as e:
                logger.error(f"✗ Error saving {kernel_name}: {e}")
                results.append({
                    "kernel_name": kernel_name,
                    "torch_name": name,
                    "success": False,
                    "error": f"Save error: {str(e)}",
                })
        else:
            logger.warning(f"✗ Failed to generate code for {kernel_name}")
            results.append({
                "kernel_name": kernel_name,
                "torch_name": name,
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
    logger.info(f"Total K1 Wrappers: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success Rate: {successful / total * 100:.2f}%" if total > 0 else "0%")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate PyTorch kernels from Kaldi K1 CUDA kernels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate for all K1 kernels
  python script/generate_torch_sample4k1.py --name all
  
  # Generate for a specific K1 wrapper (without cuda_ prefix)
  python script/generate_torch_sample4k1.py --name copy_low_upp
  
  # Use custom output directory
  python script/generate_sample4k1.py --name all --output-dir ./my_output
  
  # Use different model
  python script/generate_sample4k1.py --name all --server-type deepseek --model-name deepseek-coder
        """
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="all",
        help="Name of the K1 CUDA wrapper to generate (default: all). Use 'all' to generate for all K1 wrappers. Use wrapper name without 'cuda_' prefix (e.g., 'copy_low_upp' not 'cuda_copy_low_upp')."
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
    run_name = f"torch_{args.model_name}_num_samples_{args.num_samples}_temp_{args.temperature}_max_tokens_{args.max_tokens}_{today()}"
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
