#!/usr/bin/env python3
"""
Generate test_func for vLLM operators using ksyun API.

Usage:
    python scripts/generate_vllm_testfunc.py --op-name moe_align_block_size --output-dir output_vllm_testfunc
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator.vllm.vllm_test_prompt_generator import build_vllm_testfunc_prompt


def main():
    parser = argparse.ArgumentParser(description="Generate vLLM test functions")
    parser.add_argument("--op-name", type=str, required=True, help="Operator name (e.g., moe_align_block_size)")
    parser.add_argument("--output-dir", type=str, default="output_vllm_testfunc", help="Output directory")
    parser.add_argument("--save-prompt", action="store_true", help="Save prompt to file")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate prompt
    print(f"Generating prompt for operator: {args.op_name}")
    prompt = build_vllm_testfunc_prompt(args.op_name)

    # Save prompt if requested
    if args.save_prompt:
        prompt_file = output_dir / f"prompt_{args.op_name}.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        print(f"Prompt saved to: {prompt_file}")

    # Print prompt info
    print(f"\nPrompt length: {len(prompt)} characters")
    print("="*80)
    print("Prompt preview (first 500 chars):")
    print(prompt[:500])
    print("...")
    print("="*80)

    print(f"\nNext step: Use this prompt with ksyun API to generate test_func")
    print(f"Command example:")
    print(f"  export KSYUN_API_KEY='your-api-key'")
    print(f"  # Then use generator with --server-type ksyun")


if __name__ == "__main__":
    main()
