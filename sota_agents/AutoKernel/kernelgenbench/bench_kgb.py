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

"""KernelGenBench evaluation harness for AutoKernel.

Thin wrapper around verify_single.py that outputs AutoKernel-compatible
greppable lines (correctness, speedup, kernel_time_ms, reference_time_ms).

Usage:
    uv run kernelgenbench/bench_kgb.py                          # eval kernel.py
    uv run kernelgenbench/bench_kgb.py --kernel path/to/k.py    # eval specific file
    uv run kernelgenbench/bench_kgb.py --verbose                # detailed output
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PROJECT_DIR / "workspace"
KGB_ACTIVE_DIR = WORKSPACE_DIR / "kgb_active"
KERNEL_PY = PROJECT_DIR / "kernel.py"


def _find_verify_single() -> Path | None:
    """Find verify_single.py by searching known locations."""
    # 1. Check PROJECT_ROOT env var
    project_root = os.environ.get("PROJECT_ROOT")
    if project_root:
        candidate = Path(project_root) / "agent_bench" / "tools" / "verify_single.py"
        if candidate.exists():
            return candidate

    # 2. Walk up from this script
    p = SCRIPT_DIR
    for _ in range(10):
        p = p.parent
        candidate = p / "agent_bench" / "tools" / "verify_single.py"
        if candidate.exists():
            return candidate

    return None


def _load_metadata() -> dict:
    """Load active problem metadata."""
    meta_path = KGB_ACTIVE_DIR / "metadata.json"
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KernelGenBench evaluation for AutoKernel",
    )
    parser.add_argument(
        "--kernel", type=Path, default=None,
        help=f"Path to kernel file (default: {KERNEL_PY})",
    )
    parser.add_argument(
        "--op", type=str, default=None,
        help="Operator name (default: from metadata.json)",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name (default: from metadata.json)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode (no effect currently, for CLI compat)",
    )
    args = parser.parse_args()

    # Resolve kernel path
    kernel_path = args.kernel or KERNEL_PY
    if not kernel_path.exists():
        print(f"correctness: FAIL")
        print(f"speedup: 0.000x")
        print(f"ERROR: Kernel file not found: {kernel_path}")
        sys.exit(1)

    # Resolve op and dataset from metadata or args
    meta = _load_metadata()
    op_name = args.op or meta.get("op_name")
    dataset = args.dataset or meta.get("dataset", "KernelGenBench")

    if not op_name:
        print("correctness: FAIL")
        print("speedup: 0.000x")
        print("ERROR: No operator name. Use --op or set up via bridge_kgb.py")
        sys.exit(1)

    # Convert filename format (aten__add) back to namespace format (aten::add)
    if "::" not in op_name and "__" in op_name:
        ns, name = op_name.split("__", 1)
        op_name = f"{ns}::{name}"

    # Find verify_single.py
    verify_script = _find_verify_single()
    if verify_script is None:
        print("correctness: FAIL")
        print("speedup: 0.000x")
        print("ERROR: Could not find verify_single.py")
        sys.exit(1)

    # Create temporary verify directory (workaround for Pydantic inf serialization)
    verify_dir = kernel_path.parent / ".verify_tmp"
    verify_dir.mkdir(exist_ok=True)

    # Call verify_single.py
    cmd = [
        sys.executable, str(verify_script),
        "--code", str(kernel_path),
        "--op", op_name,
        "--dataset", dataset,
        "--output-dir", str(verify_dir),
        "--output-json",
    ]
    if args.verbose:
        cmd.append("--verbose")

    if args.verbose:
        print(f"[bench_kgb] Running: {' '.join(cmd)}")

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print("correctness: FAIL")
        print("speedup: 0.000x")
        print("ERROR: verify_single.py timed out (900s)")
        sys.exit(1)

    if proc.stderr and args.verbose:
        print(proc.stderr, file=sys.stderr)

    # Parse JSON result (take last non-empty line to skip any logging pollution)
    stdout_lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    try:
        result = json.loads(stdout_lines[-1]) if stdout_lines else {}
    except (json.JSONDecodeError, ValueError):
        print("correctness: FAIL")
        print("speedup: 0.000x")
        print(f"ERROR: Failed to parse verify_single.py output: {proc.stdout[:500]}")
        if proc.stderr:
            print(f"STDERR: {proc.stderr[:500]}", file=sys.stderr)
        sys.exit(1)

    passed = result.get("passed", False)
    total_tests = result.get("total_tests", 0)
    passed_tests = result.get("passed_tests", 0)
    speedup = result.get("speedup")
    error = result.get("error")

    # Output AutoKernel-compatible greppable lines
    correctness = "PASS" if passed else "FAIL"
    print(f"correctness: {correctness}")

    if speedup is not None and speedup > 0:
        ref_runtime = 1.0  # placeholder
        runtime = ref_runtime / speedup
        print(f"speedup: {speedup:.3f}x")
        print(f"kernel_time_ms: {runtime:.4f}")
        print(f"reference_time_ms: {ref_runtime:.4f}")
    else:
        print("speedup: 0.000x")
        print("kernel_time_ms: 0.0000")
        print("reference_time_ms: 0.0000")

    print(f"tests: {passed_tests}/{total_tests}")

    # fast_p thresholds
    for threshold in [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0]:
        passes = passed and speedup is not None and speedup >= threshold
        print(f"fast_{threshold}: {'PASS' if passes else 'FAIL'}")

    if error:
        print(f"ERROR: {error[:500]}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
