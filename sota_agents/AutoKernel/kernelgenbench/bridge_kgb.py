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

"""KernelGenBench Bridge -- Load and set up KernelGenBench problems for AutoKernel.

Unlike KernelBench (which uses HuggingFace datasets), KernelGenBench problems
are local operator prompts. This bridge sets up the workspace for AutoKernel
to generate or optimize standalone Triton kernels.

Usage:
    uv run kernelgenbench/bridge_kgb.py setup --op aten__add --dataset KernelGenBench
    uv run kernelgenbench/bridge_kgb.py setup --op aten__add --prompt /path/to/prompt.md
    uv run kernelgenbench/bridge_kgb.py info
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PROJECT_DIR / "workspace"
KGB_ACTIVE_DIR = WORKSPACE_DIR / "kgb_active"
KERNEL_PY = PROJECT_DIR / "kernel.py"


def _find_prompts_dir() -> Path | None:
    """Find the KernelGenBench prompts directory."""
    project_root = os.environ.get("PROJECT_ROOT")
    if project_root:
        candidate = Path(project_root) / "agent_bench" / "prompts" / "KernelGenBench"
        if candidate.exists():
            return candidate

    p = SCRIPT_DIR
    for _ in range(10):
        p = p.parent
        candidate = p / "agent_bench" / "prompts" / "KernelGenBench"
        if candidate.exists():
            return candidate

    return None


def setup_problem(
    op_name: str,
    dataset: str = "KernelGenBench",
    prompt_path: Path | None = None,
    baseline_kernel: Path | None = None,
) -> None:
    """Set up workspace for a KernelGenBench problem.

    Creates:
      workspace/kgb_active/operator_spec.md  -- operator prompt
      workspace/kgb_active/metadata.json     -- problem metadata
      kernel.py                              -- starter kernel (or baseline)
    """
    KGB_ACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Resolve prompt
    if prompt_path is None:
        prompts_dir = _find_prompts_dir()
        if prompts_dir:
            prompt_path = prompts_dir / f"{op_name}.md"

    # Copy operator spec
    if prompt_path and prompt_path.exists():
        spec_path = KGB_ACTIVE_DIR / "operator_spec.md"
        spec_path.write_text(prompt_path.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        print(f"WARNING: Prompt not found for {op_name}")

    # Convert op_name format for display
    display_name = op_name
    if "__" in op_name and "::" not in op_name:
        ns, name = op_name.split("__", 1)
        display_name = f"{ns}::{name}"

    # Write metadata
    metadata = {
        "op_name": op_name,
        "display_name": display_name,
        "dataset": dataset,
        "mode": "optimize" if baseline_kernel else "generate",
    }
    meta_path = KGB_ACTIVE_DIR / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Set up kernel.py
    if baseline_kernel and baseline_kernel.exists():
        # Optimization mode: copy baseline
        KERNEL_PY.write_text(
            baseline_kernel.read_text(encoding="utf-8"), encoding="utf-8"
        )
        mode = "optimize"
    else:
        # Generation mode: create starter
        starter = f'''"""\nKernelGenBench: {display_name}\n\nGenerate a standalone Triton kernel for this operator.\nRead workspace/kgb_active/operator_spec.md for the full specification.\n\nThe kernel must:\n1. Implement the operator using Triton\n2. Register with PyTorch dispatch (torch.library)\n3. Pass all accuracy tests\n4. Achieve speedup >= 1.0x over PyTorch native\n\nRun `uv run kernelgenbench/bench_kgb.py` to evaluate.\n"""\n\n# TODO: Implement the {display_name} operator\n# Read workspace/kgb_active/operator_spec.md for details\n\nimport torch\nimport triton\nimport triton.language as tl\n'''
        KERNEL_PY.write_text(starter, encoding="utf-8")
        mode = "generate"

    # Report
    print(f"=== KernelGenBench Problem Setup ===")
    print(f"  Operator:  {display_name}")
    print(f"  Dataset:   {dataset}")
    print(f"  Mode:      {mode}")
    print(f"  Kernel:    kernel.py  <-- EDIT THIS")
    print(f"  Spec:      workspace/kgb_active/operator_spec.md")
    print()
    print("Next steps:")
    print("  1. Read the operator spec")
    print("  2. Implement the kernel in kernel.py")
    print("  3. Run: uv run kernelgenbench/bench_kgb.py")
    print("  4. Iterate until correctness: PASS and speedup > 1.0x")


def show_info() -> None:
    """Show current active problem info."""
    meta_path = KGB_ACTIVE_DIR / "metadata.json"
    if not meta_path.exists():
        print("No active problem. Run: uv run kernelgenbench/bridge_kgb.py setup --op <name>")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"Active problem: {meta.get('display_name', meta.get('op_name', '?'))}")
    print(f"  Dataset: {meta.get('dataset', '?')}")
    print(f"  Mode:    {meta.get('mode', '?')}")
    print(f"  Kernel:  {'exists' if KERNEL_PY.exists() else 'missing'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="KernelGenBench Bridge for AutoKernel",
    )
    sub = parser.add_subparsers(dest="command")

    setup_p = sub.add_parser("setup", help="Set up workspace for a problem")
    setup_p.add_argument("--op", type=str, required=True, help="Operator name (e.g., aten__add)")
    setup_p.add_argument("--dataset", type=str, default="KernelGenBench")
    setup_p.add_argument("--prompt", type=Path, default=None, help="Path to prompt file")
    setup_p.add_argument("--baseline", type=Path, default=None, help="Baseline kernel to optimize")

    sub.add_parser("info", help="Show active problem info")

    args = parser.parse_args()

    if args.command == "setup":
        setup_problem(
            op_name=args.op,
            dataset=args.dataset,
            prompt_path=args.prompt,
            baseline_kernel=args.baseline,
        )
    elif args.command == "info":
        show_info()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
