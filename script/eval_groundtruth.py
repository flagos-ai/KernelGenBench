#!/usr/bin/env python3
"""
Evaluate generated PyTorch ground truth code for correctness.

This script reads generated PyTorch kernels from the output directory,
calls LLM API to evaluate if the code is correct and can serve as ground truth.
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def today() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from flagbench.dataset.kernel_list_k1 import IMPL_INFO_K1
from flagbench.dataset.kernel_list_k2 import IMPL_INFO_K2
from generator.sampler.utils import create_inference_server_from_presets
from generator.sampler.generate_samples import GenerationConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_evaluation_prompt(kernel_name: str, impl_info: Dict[str, Any], torch_code: str, is_k2: bool = False) -> str:
    """
    Create a prompt for LLM to evaluate the correctness of generated PyTorch code.
    
    Args:
        kernel_name: Name of the kernel (e.g., 'add_row_sum_mat' for K1, 'Equal' for K2)
        impl_info: Implementation info from IMPL_INFO_K1 or IMPL_INFO_K2
        torch_code: Generated PyTorch code to evaluate
        is_k2: Whether this is a K2 kernel (default: False for K1)
    
    Returns:
        Evaluation prompt string
    """
    kernel_type = "K2" if is_k2 else "K1"
    wrapper_prefix = "" if is_k2 else "cuda_"
    
    prompt = f"You are an expert PyTorch and CUDA programmer. Your task is to evaluate whether a generated PyTorch implementation correctly implements a {kernel_type} CUDA wrapper function.\n\n"
    
    prompt += f"## {kernel_type} CUDA Wrapper Function Specification:\n"
    prompt += f"**Function Name:** `{wrapper_prefix}{kernel_name}`\n"
    prompt += f"**Description:** {impl_info.get('description', 'N/A')}\n\n"
    
    # Add input arguments
    if "input_args" in impl_info and impl_info["input_args"]:
        prompt += "**Input Arguments:**\n"
        for arg in impl_info["input_args"]:
            arg_name = arg.get("name", "unknown")
            arg_type = arg.get("type", "unknown")
            prompt += f"  - `{arg_name}`: {arg_type}\n"
        prompt += "\n"
    
    # Add output arguments
    if "output_args" in impl_info and impl_info["output_args"]:
        prompt += "**Output:**\n"
        for arg in impl_info["output_args"]:
            arg_type = arg.get("type", "void")
            prompt += f"  - {arg_type}\n"
        prompt += "\n"
    
    prompt += "## Generated PyTorch Code to Evaluate:\n"
    prompt += "```python\n"
    prompt += torch_code
    prompt += "\n```\n\n"
    
    prompt += "## Evaluation Criteria:\n"
    prompt += "Please evaluate the generated PyTorch code based on the following criteria. Focus on whether the code CAN BE USED and whether the FUNCTIONALITY IS CORRECT, not on strict compliance with CUDA wrapper specifications.\n\n"
    prompt += "1. **Can the code run?**: Does the code have any syntax errors, API usage errors, or runtime errors that would prevent it from executing?\n"
    prompt += "   - Check for correct PyTorch API usage (e.g., correct parameter order, correct method names)\n"
    prompt += "   - Check for shape mismatches that would cause runtime errors\n"
    prompt += "   - Check for type errors or missing imports\n\n"
    prompt += "2. **Is the functionality correct?**: Does the code correctly implement the described functionality?\n"
    prompt += "   - Does the mathematical/logical operation match the description?\n"
    prompt += "   - Does it produce the expected output for typical inputs?\n"
    prompt += "   - Are in-place operations handled correctly (if applicable)?\n\n"
    prompt += "3. **Basic requirements**: Are basic requirements met for the code to work?\n"
    prompt += "   - Are tensors placed on CUDA device (if required)?\n"
    prompt += "   - Are basic dimension checks present to prevent obvious errors?\n\n"
    prompt += "**IMPORTANT**: Do NOT mark code as incorrect for:\n"
    prompt += "- Missing dtype checks (e.g., not enforcing torch.float64 for double*)\n"
    prompt += "- Not handling MatrixDim or stride parameters (unless they cause runtime errors)\n"
    prompt += "- Missing edge case handling (unless it causes the code to fail)\n"
    prompt += "- Minor deviations from CUDA wrapper specifications that don't affect functionality\n\n"
    
    prompt += "## Your Task:\n"
    prompt += "Please provide a detailed evaluation in the following format. You must first analyze each criterion and provide reasoning, then make a final judgment based on your analysis.\n\n"
    prompt += "```\n"
    prompt += "## DETAILED REASONING:\n"
    prompt += "For each evaluation criterion, provide your analysis:\n\n"
    prompt += "1. **Can the code run? Analysis:**\n"
    prompt += "   - [Check for syntax errors - can Python parse this code?]\n"
    prompt += "   - [Check for API usage errors - are PyTorch methods called correctly?]\n"
    prompt += "   - [Check for shape mismatches - will operations fail due to incompatible shapes?]\n"
    prompt += "   - [Check for type errors - are operations performed on compatible types?]\n"
    prompt += "   - [Check for missing imports or undefined variables]\n\n"
    prompt += "2. **Is the functionality correct? Analysis:**\n"
    prompt += "   - [Does the code implement the described mathematical/logical operation?]\n"
    prompt += "   - [For typical inputs, would it produce the expected output?]\n"
    prompt += "   - [Are in-place operations handled correctly (if applicable)?]\n"
    prompt += "   - [Does the logic match the function description?]\n\n"
    prompt += "3. **Basic Requirements Analysis:**\n"
    prompt += "   - [Are tensors on CUDA device (if required)?]\n"
    prompt += "   - [Are there basic dimension checks to prevent obvious errors?]\n"
    prompt += "   - [Are there any critical missing checks that would cause runtime failures?]\n\n"
    prompt += "## FINAL JUDGMENT:\n"
    prompt += "Based on the above analysis, provide your final judgment:\n\n"
    prompt += "IS_CORRECT: [YES/NO]\n"
    prompt += "   - YES if: The code can run without errors AND the functionality is correct\n"
    prompt += "   - NO if: The code has runtime errors OR the functionality is incorrect\n\n"
    prompt += "CONFIDENCE: [HIGH/MEDIUM/LOW]\n"
    prompt += "JUDGMENT_REASONING: [Brief summary of why you reached this conclusion, focusing on whether the code can be used and if the functionality is correct]\n\n"
    prompt += "ISSUES:\n"
    prompt += "- [List any critical issues that prevent the code from running OR make the functionality incorrect]\n"
    prompt += "- [Do NOT list issues about missing dtype checks, MatrixDim handling, or stride parameters unless they cause runtime errors]\n"
    prompt += "- [Write 'None' if there are no critical issues]\n\n"
    prompt += "SUGGESTIONS:\n"
    prompt += "- [List any suggestions for improvement (optional enhancements, not requirements)]\n"
    prompt += "- [Write 'None' if no improvements needed]\n"
    prompt += "```\n\n"
    prompt += "**Important Notes:**\n"
    prompt += "- Focus on PRACTICAL USABILITY: Can this code be used? Does it work correctly?\n"
    prompt += "- Do NOT penalize code for missing strict CUDA wrapper compliance (dtype checks, MatrixDim, stride handling) unless they cause actual runtime errors.\n"
    prompt += "- A code that runs correctly and implements the right functionality should be marked as CORRECT, even if it doesn't match every detail of the CUDA wrapper specification.\n"
    prompt += "- Provide detailed reasoning for each criterion before making your final judgment.\n"
    prompt += "- Your final judgment should be based on whether the code CAN BE USED and whether the FUNCTIONALITY IS CORRECT."
    
    return prompt


def evaluate_single_code(
    kernel_name: str,
    impl_info: Dict[str, Any],
    torch_code: str,
    inference_server: callable,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    is_k2: bool = False
) -> Dict[str, Any]:
    """
    Evaluate a single PyTorch code using LLM with retry logic.
    
    Args:
        kernel_name: Name of the kernel (K1 or K2)
        impl_info: Implementation info from IMPL_INFO_K1 or IMPL_INFO_K2
        torch_code: Generated PyTorch code to evaluate
        inference_server: LLM inference server function
        max_retries: Maximum number of retry attempts (default: 3)
        retry_delay: Initial delay between retries in seconds (default: 1.0, uses exponential backoff)
        is_k2: Whether this is a K2 kernel (default: False for K1)
    
    Returns:
        Evaluation result dictionary
    """
    import time
    
    prompt = create_evaluation_prompt(kernel_name, impl_info, torch_code, is_k2=is_k2)
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"Retrying evaluation for {kernel_name} (attempt {attempt + 1}/{max_retries + 1})...")
            else:
                logger.info(f"Evaluating {kernel_name}...")
            
            response = inference_server(prompt)
            
            # Parse response
            evaluation_text = response[0] if isinstance(response, list) else response
            
            # Try to extract structured information
            is_correct = "IS_CORRECT: YES" in evaluation_text.upper()
            confidence = "HIGH"
            if "CONFIDENCE: MEDIUM" in evaluation_text.upper():
                confidence = "MEDIUM"
            elif "CONFIDENCE: LOW" in evaluation_text.upper():
                confidence = "LOW"
            
            # Extract judgment reasoning if available
            judgment_reasoning = ""
            if "JUDGMENT_REASONING:" in evaluation_text:
                # Extract the reasoning section
                reasoning_start = evaluation_text.find("JUDGMENT_REASONING:")
                reasoning_end = evaluation_text.find("\n\n", reasoning_start)
                if reasoning_end == -1:
                    reasoning_end = len(evaluation_text)
                judgment_reasoning = evaluation_text[reasoning_start:reasoning_end].replace("JUDGMENT_REASONING:", "").strip()
            
            return {
                "kernel_name": kernel_name,
                "is_correct": is_correct,
                "confidence": confidence,
                "judgment_reasoning": judgment_reasoning,
                "evaluation_text": evaluation_text,
                "success": True,
            }
        except Exception as e:
            error_msg = str(e).lower()
            is_retryable = any(keyword in error_msg for keyword in [
                "connection", "timeout", "gateway", "503", "504", "502", "500"
            ])
            
            if attempt < max_retries and is_retryable:
                # Exponential backoff: delay = retry_delay * (2 ^ attempt)
                delay = retry_delay * (2 ** attempt)
                logger.warning(
                    f"Retryable error for {kernel_name} (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
            else:
                logger.error(f"Error evaluating {kernel_name}: {e}", exc_info=True)
                return {
                    "kernel_name": kernel_name,
                    "is_correct": False,
                    "confidence": "LOW",
                    "evaluation_text": f"Error during evaluation: {str(e)}",
                    "success": False,
                    "error": str(e),
                }
    
    # Should not reach here, but just in case
    return {
        "kernel_name": kernel_name,
        "is_correct": False,
        "confidence": "LOW",
        "evaluation_text": "Max retries exceeded",
        "success": False,
        "error": "Max retries exceeded",
    }


def evaluate_groundtruth(
    input_dir: Path,
    output_dir: Path,
    config: GenerationConfig,
    num_workers: int = 10,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    name: str = "all"
) -> None:
    """
    Evaluate all generated PyTorch ground truth codes.
    
    Args:
        input_dir: Directory containing generated PyTorch codes
        output_dir: Directory to save evaluation results
        config: Generation configuration (for API settings)
        num_workers: Number of parallel workers for evaluation
    """
    # Find all groundtruth directories
    groundtruth_dirs = list(input_dir.glob("groundtruth_*"))
    if not groundtruth_dirs:
        logger.warning(f"No groundtruth_* directories found in {input_dir}")
        return
    
    # Use the first groundtruth directory
    groundtruth_dir = groundtruth_dirs[0]
    logger.info(f"Evaluating codes from: {groundtruth_dir}")
    
    # Find all PyTorch files
    torch_files = list(groundtruth_dir.glob("torch_*.py"))
    if not torch_files:
        logger.warning(f"No torch_*.py files found in {groundtruth_dir}")
        return
    
    logger.info(f"Found {len(torch_files)} PyTorch files to evaluate")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference server
    inference_server = create_inference_server_from_presets(
        server_type=config.server_type,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        verbose=config.verbose,
    )
    
    # Prepare evaluation tasks
    evaluation_tasks = []
    for torch_file in torch_files:
        # Extract kernel name from filename (remove 'torch_' prefix and '.py' suffix)
        kernel_name = torch_file.stem.replace("torch_", "")
        
        # Check if it's K1 or K2
        is_k2 = False
        impl_info = None
        
        if kernel_name in IMPL_INFO_K2:
            impl_info = IMPL_INFO_K2[kernel_name]
            is_k2 = True
            logger.debug(f"Kernel {kernel_name} found in IMPL_INFO_K2 (K2)")
        elif kernel_name in IMPL_INFO_K1:
            impl_info = IMPL_INFO_K1[kernel_name]
            is_k2 = False
            logger.debug(f"Kernel {kernel_name} found in IMPL_INFO_K1 (K1)")
        else:
            logger.warning(f"Kernel {kernel_name} not found in IMPL_INFO_K1 or IMPL_INFO_K2, skipping")
            continue
        
        # Read PyTorch code
        try:
            with open(torch_file, "r") as f:
                torch_code = f.read()
        except Exception as e:
            logger.error(f"Error reading {torch_file}: {e}")
            continue
        
        evaluation_tasks.append((kernel_name, impl_info, torch_code, is_k2))
    
    # Filter by name if specified
    if name and name.lower() != "all":
        evaluation_tasks = [(k, i, c, is_k2) for k, i, c, is_k2 in evaluation_tasks if k == name]
        if not evaluation_tasks:
            logger.error(f"Kernel '{name}' not found in the groundtruth directory")
            return
        logger.info(f"Filtered to {len(evaluation_tasks)} kernel(s) matching '{name}'")
    
    # Evaluate in parallel
    logger.info(f"Evaluating {len(evaluation_tasks)} kernels with {num_workers} workers...")
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                evaluate_single_code,
                kernel_name,
                impl_info,
                torch_code,
                inference_server,
                max_retries=max_retries,
                retry_delay=retry_delay,
                is_k2=is_k2
            ): kernel_name
            for kernel_name, impl_info, torch_code, is_k2 in evaluation_tasks
        }
        
        for future in as_completed(futures):
            kernel_name = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = "✓" if result["is_correct"] else "✗"
                logger.info(f"{status} {kernel_name}: {result['confidence']} confidence")
            except Exception as e:
                logger.error(f"Error evaluating {kernel_name}: {e}")
                results.append({
                    "kernel_name": kernel_name,
                    "is_correct": False,
                    "confidence": "LOW",
                    "evaluation_text": f"Error: {str(e)}",
                    "success": False,
                    "error": str(e),
                })
    
    # Calculate statistics
    total = len(results)
    correct = sum(1 for r in results if r.get("is_correct", False))
    incorrect = total - correct
    high_confidence = sum(1 for r in results if r.get("confidence") == "HIGH")
    
    # Save detailed results
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump({
            "total": total,
            "correct": correct,
            "incorrect": incorrect,
            "correct_rate": f"{correct / total * 100:.2f}%" if total > 0 else "0%",
            "high_confidence": high_confidence,
            "results": results,
        }, f, indent=2)
    
    # Save summary
    summary_path = output_dir / "evaluation_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("Ground Truth Evaluation Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Evaluated: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Incorrect: {incorrect}\n")
        f.write(f"Correct Rate: {correct / total * 100:.2f}%\n" if total > 0 else "0%\n")
        f.write(f"High Confidence: {high_confidence}\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("Detailed Results:\n")
        f.write("=" * 60 + "\n\n")
        
        for result in results:
            status = "✓" if result.get("is_correct", False) else "✗"
            f.write(f"{status} {result['kernel_name']} ({result.get('confidence', 'UNKNOWN')})\n")
            
            # Show judgment reasoning if available (full content)
            if result.get("judgment_reasoning"):
                f.write(f"  Judgment Reasoning: {result['judgment_reasoning']}\n\n")
            
            # Show full evaluation text for detailed review
            if result.get("evaluation_text"):
                f.write("  Full Evaluation:\n")
                f.write("  " + "=" * 58 + "\n")
                # Indent each line for readability
                evaluation_lines = result['evaluation_text'].split('\n')
                for line in evaluation_lines:
                    f.write(f"  {line}\n")
                f.write("  " + "=" * 58 + "\n")
            
            f.write("\n")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluation Summary:")
    logger.info(f"Total Evaluated: {total}")
    logger.info(f"Correct: {correct}")
    logger.info(f"Incorrect: {incorrect}")
    logger.info(f"Correct Rate: {correct / total * 100:.2f}%" if total > 0 else "0%")
    logger.info(f"High Confidence: {high_confidence}")
    logger.info(f"Results saved to: {results_path}")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate generated PyTorch ground truth code for correctness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all generated codes in a directory
  python script/eval_groundtruth.py --input-dir output/torch_deepseek-v3-0324_...
  
  # Evaluate a specific kernel only
  python script/eval_groundtruth.py --input-dir output/torch_deepseek-v3-0324_... --name add_row_sum_mat
  
  # Use specific model for evaluation
  python script/eval_groundtruth.py --input-dir output/torch_deepseek-v3-0324_... --model-name gpt-5-2025-08-07
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing generated PyTorch codes (should contain groundtruth_* subdirectories)"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        default="all",
        help="Name of the kernel to evaluate (default: all). Use 'all' to evaluate all kernels, or specify a kernel name (e.g., 'add_row_sum_mat' for K1, 'Equal' for K2)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for evaluation results (default: input_dir/eval_<timestamp>)"
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
        default="gpt-5-2025-08-07",
        help="Model name to use for evaluation (default: gpt-5-2025-08-07)"
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
        default=4096,
        help="Maximum tokens to generate (default: 4096)"
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
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.input_dir / f"eval_{today()}"
    
    # Create generation config
    config = GenerationConfig(
        run_name="eval_groundtruth",
        server_type=args.server_type,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_workers=args.num_workers,
        num_samples=1,
        verbose=args.verbose,
        run_dir=str(args.output_dir),
        log_prompt=False,
    )
    
    logger.info("Starting ground truth evaluation...")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Model: {args.model_name} ({args.server_type})")
    logger.info(f"Config: {config}")
    
    # Evaluate
    evaluate_groundtruth(args.input_dir, args.output_dir, config, args.num_workers, args.max_retries, args.retry_delay, args.name)
    
    logger.info("Evaluation completed!")


if __name__ == "__main__":
    main()

