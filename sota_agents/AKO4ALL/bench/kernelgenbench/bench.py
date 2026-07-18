#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""KernelGenBench evaluation wrapper for AKO4ALL.

Thin wrapper around verify_single.py that outputs AKO4ALL-compatible
structured lines (COMPILED, CORRECT, RUNTIME, REF_RUNTIME, SPEEDUP, TESTS).

Usage:
    python bench/kernelgenbench/bench.py --solution solution/kernel.py --op aten__add --dataset KernelGenBench --verbose
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# verify_single.py lives at: <project_root>/agent_bench/tools/verify_single.py
# This script lives at:      <project_root>/sota_agents/AKO4ALL/bench/kernelgenbench/bench.py
# But at runtime, the workspace is a copy, so we resolve relative to PROJECT_ROOT env or search upward.
SCRIPT_DIR = Path(__file__).resolve().parent
VERIFY_SINGLE = None

def _find_verify_single():
    """Find verify_single.py by searching known locations."""
    # 1. Check PROJECT_ROOT env var
    import os
    project_root = os.environ.get("PROJECT_ROOT")
    if project_root:
        candidate = Path(project_root) / "agent_bench" / "tools" / "verify_single.py"
        if candidate.exists():
            return candidate

    # 2. Walk up from this script to find agent_bench/tools/verify_single.py
    p = SCRIPT_DIR
    for _ in range(10):
        p = p.parent
        candidate = p / "agent_bench" / "tools" / "verify_single.py"
        if candidate.exists():
            return candidate

    return None


def main():
    parser = argparse.ArgumentParser(description="KernelGenBench benchmark for AKO4ALL")
    parser.add_argument("--solution", type=Path, required=True, help="Path to kernel file")
    parser.add_argument("--op", type=str, required=True, help="Operator name")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    verify_script = _find_verify_single()
    if verify_script is None:
        print("COMPILED: False")
        print("CORRECT: False")
        print("ERROR: Could not find verify_single.py")
        sys.exit(1)

    # Convert filename format (aten__add) back to namespace format (aten::add)
    op_name = args.op
    if "::" not in op_name and "__" in op_name:
        ns, name = op_name.split("__", 1)
        op_name = f"{ns}::{name}"

    if not args.solution.exists():
        print("COMPILED: False")
        print("CORRECT: False")
        print(f"ERROR: Solution file not found: {args.solution}")
        sys.exit(1)

    # Create temporary verify directory to enable fallback speedup reading
    # This works around Pydantic serialization bug where float('inf') becomes null
    verify_dir = args.solution.parent / ".verify_tmp"
    verify_dir.mkdir(exist_ok=True)

    # Call verify_single.py with --output-dir and --output-json
    cmd = [
        sys.executable, str(verify_script),
        "--code", str(args.solution),
        "--op", op_name,
        "--dataset", args.dataset,
        "--output-dir", str(verify_dir),
        "--output-json",
    ]
    if args.verbose:
        cmd.append("--verbose")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print("COMPILED: False")
        print("CORRECT: False")
        print("ERROR: verify_single.py timed out")
        sys.exit(1)

    # Print stderr (verbose logs) if present
    if proc.stderr and args.verbose:
        print(proc.stderr, file=sys.stderr)

    # Parse JSON result from stdout
    try:
        result = json.loads(proc.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        print("COMPILED: False")
        print("CORRECT: False")
        print(f"ERROR: Failed to parse verify_single.py output: {proc.stdout[:500]}")
        if proc.stderr:
            print(f"STDERR: {proc.stderr[:500]}", file=sys.stderr)
        sys.exit(1)

    passed = result.get("passed", False)
    total_tests = result.get("total_tests", 0)
    passed_tests = result.get("passed_tests", 0)
    speedup = result.get("speedup")
    error = result.get("error")

    # Output AKO4ALL-compatible structured lines
    compiled = total_tests > 0 or passed
    print(f"COMPILED: {compiled}")
    print(f"CORRECT: {passed}")

    if speedup is not None and speedup > 0:
        # Estimate runtimes from speedup ratio (verify_single doesn't give absolute times)
        # Use placeholder values; the speedup ratio is what matters
        ref_runtime = 1.0
        runtime = ref_runtime / speedup
        print(f"RUNTIME: {runtime:.4f}")
        print(f"REF_RUNTIME: {ref_runtime:.4f}")
        print(f"SPEEDUP: {speedup:.4f}x")
    else:
        print("RUNTIME: N/A")
        print("REF_RUNTIME: N/A")
        print("SPEEDUP: N/A")

    print(f"TESTS: {passed_tests}/{total_tests}")

    if error:
        print(f"ERROR: {error[:500]}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
