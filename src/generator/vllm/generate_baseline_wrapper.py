#!/usr/bin/env python3
"""
Generate vLLM baseline wrapper from operator signatures.

Usage:
    # Generate single operator:
    python src/generator/vllm/generate_baseline_wrapper.py --op-name rotary_embedding

    # Generate multiple operators:
    python src/generator/vllm/generate_baseline_wrapper.py --op-name rotary_embedding,fused_add_rms_norm,silu_and_mul

    # Generate all operators in JSON:
    python src/generator/vllm/generate_baseline_wrapper.py --all
"""

import argparse
import json
import re
import sys
from pathlib import Path

SIGNATURES_PATH = Path(__file__).with_name("vllm_ops_signatures_split.json")
OUTPUT_DIR = Path(__file__).parents[2] / "flagbench" / "dataset" / "baseline" / "vllm"


def load_signatures(path: Path = SIGNATURES_PATH):
    return json.loads(path.read_text(encoding="utf-8"))


def parse_signature(sig_str: str):
    """Parse signature string into (params_str, return_type).

    Returns:
        (params_str, return_type) where params_str is the raw parameter
        string and return_type is e.g. 'None', 'tuple', 'torch.Tensor'.
    """
    # Extract return type from "(...) -> XXX"
    m = re.search(r'\)\s*->\s*(.+)$', sig_str)
    return_type = m.group(1).strip() if m else None

    # Extract parameter string between outermost parens
    start = sig_str.index('(')
    # Find matching close paren (handle nested parens like tuple[int, int])
    depth = 0
    end = start
    for i in range(start, len(sig_str)):
        if sig_str[i] == '(':
            depth += 1
        elif sig_str[i] == ')':
            depth -= 1
            if depth == 0:
                end = i
                break
    params_str = sig_str[start + 1:end]
    return params_str, return_type


def extract_param_names(params_str: str):
    """Extract parameter names from a signature parameter string.

    Handles complex types like 'tuple[int, int]', 'torch.Tensor | None',
    and default values.
    """
    params = []
    current = ""
    depth = 0

    for ch in params_str:
        if ch in ('(', '['):
            depth += 1
            current += ch
        elif ch in (')', ']'):
            depth -= 1
            current += ch
        elif ch == ',' and depth == 0:
            params.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        params.append(current.strip())

    result = []
    for p in params:
        if not p:
            continue
        # "name: type = default" -> extract name
        name = p.split(':')[0].strip()
        if not name:
            continue
        # Check if it has a default value
        has_default = '=' in p
        result.append((name, has_default, p))

    return result


def generate_wrapper(op_name: str, schema: dict) -> str:
    """Generate baseline wrapper code for a single operator."""
    sig_str = schema["signature"]
    output_params = schema.get("output_parameters", "")
    doc = schema.get("doc", "")

    params_str, return_type = parse_signature(sig_str)
    param_list = extract_param_names(params_str)

    # Build function signature
    func_params = []
    for name, has_default, raw in param_list:
        # Extract type annotation and default
        parts = raw.split(':', 1)
        if len(parts) == 2:
            type_and_default = parts[1].strip()
            func_params.append(f"    {name}: {type_and_default}")
        else:
            func_params.append(f"    {name}")

    func_sig = ",\n".join(func_params)

    # Build call arguments (just parameter names)
    call_args = []
    for name, _, _ in param_list:
        call_args.append(f"        {name}")
    call_args_str = ",\n".join(call_args)

    # Determine return annotation
    if return_type and return_type != "None":
        ret_annotation = f" -> {return_type}"
        call_prefix = "return "
    else:
        ret_annotation = " -> None"
        call_prefix = ""

    # Build docstring
    if doc:
        # Clean up doc: take first 2 lines
        doc_lines = doc.strip().split('\n')
        short_doc = doc_lines[0].strip()
        if not short_doc:
            short_doc = doc_lines[1].strip() if len(doc_lines) > 1 else ""
        docstring = f'    """{short_doc}"""'
    else:
        docstring = f'    """Wrapper for vLLM {op_name} implementation."""'

    # Assemble code
    lines = [
        '"""',
        f"vLLM {op_name} baseline wrapper.",
        '"""',
        "import torch",
        "from vllm import _custom_ops",
        "",
        "",
        f"def {op_name}(",
        func_sig,
        f"){ret_annotation}:",
        docstring,
        f"    {call_prefix}_custom_ops.{op_name}(",
        call_args_str,
        "    )",
        "",
    ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate vLLM baseline wrappers from signatures JSON"
    )
    parser.add_argument(
        "--op-name",
        type=str,
        default=None,
        help="Operator name(s), comma-separated. e.g. 'rotary_embedding,rms_norm'",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate wrappers for all operators in JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated code without writing files",
    )
    args = parser.parse_args()

    if not args.op_name and not args.all:
        parser.error("Must specify --op-name or --all")

    data = load_signatures()
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        op_names = sorted(data.keys())
    else:
        op_names = [n.strip() for n in args.op_name.split(",") if n.strip()]

    success = 0
    failed = 0
    for op_name in op_names:
        if op_name not in data:
            print(f"[SKIP] {op_name}: not found in signatures JSON")
            failed += 1
            continue

        schema = data[op_name]
        if not schema.get("callable", False):
            print(f"[SKIP] {op_name}: not callable")
            failed += 1
            continue

        try:
            code = generate_wrapper(op_name, schema)
        except Exception as e:
            print(f"[FAIL] {op_name}: {e}")
            failed += 1
            continue

        if args.dry_run:
            print(f"=== {op_name} ===")
            print(code)
            print()
        else:
            out_path = output_dir / f"{op_name}.py"
            out_path.write_text(code, encoding="utf-8")
            print(f"[OK] {op_name} -> {out_path}")

        success += 1

    print(f"\nDone: {success} generated, {failed} skipped/failed")


if __name__ == "__main__":
    main()
