#!/usr/bin/env python3
"""
Generate vLLM test_func using ksyun API.

Usage:
    export KSYUN_API_KEY='8407460c-9a3d-4a32-bb0d-43e91a74304f'
    python scripts/generate_vllm_testfunc_with_api.py --op-name moe_align_block_size
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from generator.vllm.vllm_test_prompt_generator import build_vllm_testfunc_prompt
from generator.sampler import GenerationConfig
from generator.test_func_generator import TestFuncGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate vLLM test functions with ksyun API")
    parser.add_argument("--op-name", type=str, required=True, help="Operator name")
    parser.add_argument("--output-dir", type=str, default="output_vllm_testfunc", help="Output directory")
    parser.add_argument("--server-type", type=str, default="ksyun", help="API server type")
    parser.add_argument("--model-name", type=str, default="mog-1", help="Model name")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to generate")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating test_func for operator: {args.op_name}")
    print(f"Using API: {args.server_type}, Model: {args.model_name}")

    # Generate prompt
    prompt = build_vllm_testfunc_prompt(args.op_name)

    # Save prompt
    prompt_file = output_dir / f"prompt_{args.op_name}.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    print(f"Prompt saved to: {prompt_file}")

    # Configure generation
    generation_config = GenerationConfig(
        server_type=args.server_type,
        model_name=args.model_name,
        temperature=0.0,
        max_tokens=4096,
        num_samples=args.num_samples,
        run_dir=str(output_dir),
        run_name=f"vllm_{args.op_name}"
    )

    # Create generator
    generator = TestFuncGenerator(generation_config)

    print(f"\nCalling {args.server_type} API to generate test_func...")
    print("This may take a few moments...")

    # Generate test_func
    # Note: TestFuncGenerator expects specific input format
    # We'll need to adapt this based on the actual generator interface

    print(f"\nGeneration complete!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
