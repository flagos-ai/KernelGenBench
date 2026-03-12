#!/usr/bin/env python3
"""
Generate vLLM test_func using ksyun API.

Usage:
    export KSYUN_API_KEY='8407460c-9a3d-4a32-bb0d-43e91a74304f'

    # Single operator:
    python src/generator/vllm/generate_testfunc.py --op-name moe_align_block_size

    # Multiple operators (concurrent):
    python src/generator/vllm/generate_testfunc.py --op-name rotary_embedding,fused_add_rms_norm,silu_and_mul_scaled_fp4_experts_quant

    # Use GPT-5 (mog1):
    python src/generator/vllm/generate_testfunc.py --op-name rotary_embedding --model mog1
"""

import argparse
import sys
import json
import os
import asyncio
import aiohttp
from pathlib import Path
from datetime import datetime


async def call_ksyun_api(prompt: str, api_key: str, model: str = "mog1",
                        max_tokens: int = 16000) -> str:
    """调用金山云API"""
    base_url = "https://kspmas.ksyun.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_completion_tokens": max_tokens
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(base_url, headers=headers, json=payload) as response:
            if response.status == 200:
                result = await response.json()
                content = result['choices'][0]['message'].get('content')
                if not content:
                    # 尝试获取 reasoning_content
                    content = result['choices'][0]['message'].get('reasoning_content')
                if not content:
                    raise Exception(f"API returned empty content: {result}")
                return content
            else:
                error_text = await response.text()
                raise Exception(f"API error {response.status}: {error_text}")


def extract_code(text: str) -> str:
    """提取代码块"""
    if "```python" in text:
        start = text.find("```python") + 9
        end = text.find("```", start)
        if end > start:
            return text[start:end].strip()
    return text.strip()


async def generate_one_op(op_name: str, api_key: str, model: str,
                          max_tokens: int, output_root: Path,
                          build_prompt_fn) -> dict:
    """Generate test_func for a single operator."""
    output_base = output_root / op_name
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"[{op_name}] Building prompt...")
    prompt = build_prompt_fn(op_name)

    prompt_file = output_base / "prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")

    print(f"[{op_name}] Calling API (model={model})...")
    try:
        raw = await call_ksyun_api(prompt, api_key, model, max_tokens)
        code = extract_code(raw)

        code_file = output_base / f"test_{op_name}.py"
        code_file.write_text(code, encoding="utf-8")

        raw_file = output_base / "raw_response.txt"
        raw_file.write_text(raw, encoding="utf-8")

        print(f"[{op_name}] OK  ({len(code)} chars)")
        return {"op": op_name, "status": "ok", "file": str(code_file)}
    except Exception as e:
        print(f"[{op_name}] FAIL: {e}")
        return {"op": op_name, "status": "fail", "error": str(e)}


async def main_async():
    parser = argparse.ArgumentParser(
        description="Generate vLLM test functions with ksyun API"
    )
    parser.add_argument(
        "--op-name", type=str, required=True,
        help="Operator name(s), comma-separated"
    )
    parser.add_argument("--model", type=str, default="mog1", help="Model name")
    parser.add_argument("--max-tokens", type=int, default=16000)

    args = parser.parse_args()

    api_key = os.environ.get("KSYUN_API_KEY")
    if not api_key:
        print("Error: KSYUN_API_KEY not set")
        sys.exit(1)

    # Direct import to avoid generator/__init__.py dependency chain
    sys.path.insert(0, str(Path(__file__).parent))
    from vllm_test_prompt_generator import build_vllm_testfunc_prompt

    op_names = [n.strip() for n in args.op_name.split(",") if n.strip()]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(__file__).parent / "output" / timestamp

    print(f"Operators: {op_names}")
    print(f"Model: {args.model}, max_tokens: {args.max_tokens}")
    print(f"Output: {output_root}")
    print(f"Concurrency: {len(op_names)}")
    print("=" * 60)

    tasks = [
        generate_one_op(op, api_key, args.model, args.max_tokens,
                        output_root, build_vllm_testfunc_prompt)
        for op in op_names
    ]
    results = await asyncio.gather(*tasks)

    print("\n" + "=" * 60)
    ok = sum(1 for r in results if r["status"] == "ok")
    fail = sum(1 for r in results if r["status"] == "fail")
    print(f"Done: {ok} ok, {fail} fail")
    for r in results:
        status = "OK" if r["status"] == "ok" else f"FAIL: {r.get('error','')}"
        print(f"  {r['op']}: {status}")

    meta_file = output_root / "summary.json"
    meta_file.write_text(json.dumps(results, indent=2), encoding="utf-8")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
