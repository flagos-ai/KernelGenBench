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

"""
KernelBench Bridge -- Load, cache, and set up KernelBench problems for AutoKernel.

Supports three problem sources:
  1. HuggingFace dataset (ScalingIntelligence/KernelBench)
  2. Local KernelBench repo clone
  3. Individual .py problem files

Usage:
    uv run kernelbench/bridge.py setup --level 1 --problem 1
    uv run kernelbench/bridge.py setup --level 1 --problem 1 --source hf
    uv run kernelbench/bridge.py fetch --source hf --level 1
    uv run kernelbench/bridge.py fetch --source local --repo-path /path/to/KernelBench
    uv run kernelbench/bridge.py list --level 1
    uv run kernelbench/bridge.py info --level 1 --problem 1
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent  # autoresearch/
WORKSPACE_DIR = PROJECT_DIR / "workspace"
KB_CACHE_DIR = WORKSPACE_DIR / "kb_cache"
KB_ACTIVE_DIR = WORKSPACE_DIR / "kb_active"
KERNEL_PY = PROJECT_DIR / "kernel.py"


# ---------------------------------------------------------------------------
# Problem data structure
# ---------------------------------------------------------------------------

@dataclass
class KernelBenchProblem:
    """A single KernelBench problem."""

    level: int
    problem_id: int
    name: str
    source_code: str

    @property
    def uid(self) -> str:
        return f"L{self.level}_P{self.problem_id:03d}"

    @property
    def cache_path(self) -> Path:
        return KB_CACHE_DIR / f"level{self.level}" / f"{self.problem_id}.py"

    # ----- Cache persistence -----

    def save_to_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(self.source_code, encoding="utf-8")
        meta_path = self.cache_path.with_suffix(".json")
        meta = {
            "level": self.level,
            "problem_id": self.problem_id,
            "name": self.name,
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load_from_cache(cls, level: int, problem_id: int) -> Optional["KernelBenchProblem"]:
        cache_path = KB_CACHE_DIR / f"level{level}" / f"{problem_id}.py"
        meta_path = cache_path.with_suffix(".json")
        if not cache_path.exists():
            return None
        source = cache_path.read_text(encoding="utf-8")
        name = f"problem_{problem_id}"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            name = meta.get("name", name)
        return cls(level=level, problem_id=problem_id, name=name, source_code=source)

    # ----- Analysis -----

    def analyze(self) -> Dict[str, Any]:
        """Identify operations, shapes, parameter usage, and estimate difficulty."""
        analysis: Dict[str, Any] = {
            "operations": [],
            "estimated_difficulty": "unknown",
            "has_parameters": False,
            "input_shapes": [],
            "forward_lines": 0,
        }

        try:
            ast.parse(self.source_code)
        except SyntaxError:
            return analysis

        # Detect common ops by pattern matching
        OP_PATTERNS: Dict[str, str] = {
            "torch.matmul": "matmul",
            "torch.mm": "matmul",
            "torch.bmm": "batched_matmul",
            "F.linear": "linear",
            "nn.Linear": "linear",
            "F.relu": "relu",
            "F.gelu": "gelu",
            "F.silu": "silu",
            "F.softmax": "softmax",
            "F.layer_norm": "layernorm",
            "nn.LayerNorm": "layernorm",
            "F.group_norm": "groupnorm",
            "nn.GroupNorm": "groupnorm",
            "F.conv2d": "conv2d",
            "nn.Conv2d": "conv2d",
            "F.conv1d": "conv1d",
            "nn.Conv1d": "conv1d",
            "nn.ConvTranspose2d": "conv_transpose",
            "F.batch_norm": "batchnorm",
            "nn.BatchNorm2d": "batchnorm",
            "nn.BatchNorm1d": "batchnorm",
            "F.instance_norm": "instancenorm",
            "nn.InstanceNorm2d": "instancenorm",
            "F.cross_entropy": "cross_entropy",
            "F.mse_loss": "mse_loss",
            "F.nll_loss": "nll_loss",
            "torch.sum": "reduce_sum",
            "torch.mean": "reduce_mean",
            "torch.max": "reduce_max",
            "torch.sigmoid": "sigmoid",
            "torch.tanh": "tanh",
            "F.dropout": "dropout",
            "nn.Dropout": "dropout",
            "F.max_pool2d": "maxpool",
            "nn.MaxPool2d": "maxpool",
            "F.avg_pool2d": "avgpool",
            "nn.AvgPool2d": "avgpool",
            "F.interpolate": "interpolate",
            "torch.cat": "concat",
            "torch.stack": "stack",
            "F.embedding": "embedding",
            "nn.Embedding": "embedding",
            "torch.cumsum": "cumsum",
            "torch.sort": "sort",
            "torch.einsum": "einsum",
            "F.scaled_dot_product_attention": "sdpa",
        }

        for pattern, op_name in OP_PATTERNS.items():
            if pattern in self.source_code:
                if op_name not in analysis["operations"]:
                    analysis["operations"].append(op_name)

        # Check for parameters
        param_indicators = [
            "nn.Linear", "nn.Conv", "nn.BatchNorm", "nn.LayerNorm",
            "nn.GroupNorm", "nn.InstanceNorm", "nn.Embedding",
            "nn.Parameter", "self.weight", "self.bias",
        ]
        for ind in param_indicators:
            if ind in self.source_code:
                analysis["has_parameters"] = True
                break

        # Count forward() body lines (proxy for complexity)
        fwd_match = re.search(
            r'def\s+forward\s*\([^)]*\)\s*(?:->[^:]*)?:\s*\n((?:\s+.*\n)*)',
            self.source_code,
        )
        if fwd_match:
            body = fwd_match.group(1)
            non_empty = [l for l in body.split("\n") if l.strip() and not l.strip().startswith("#")]
            analysis["forward_lines"] = len(non_empty)

        # Estimate difficulty
        n_ops = len(analysis["operations"])
        fwd_lines = analysis["forward_lines"]
        if n_ops <= 1 and fwd_lines <= 3:
            analysis["estimated_difficulty"] = "easy"
        elif n_ops <= 3 and fwd_lines <= 10:
            analysis["estimated_difficulty"] = "medium"
        elif n_ops <= 6 and fwd_lines <= 25:
            analysis["estimated_difficulty"] = "hard"
        else:
            analysis["estimated_difficulty"] = "very_hard"

        # Extract input shapes from get_inputs()
        shape_re = r"torch\.(?:randn|rand|zeros|ones|empty)\s*\(([^)]+)\)"
        for m in re.finditer(shape_re, self.source_code):
            dims = re.findall(r"\d+", m.group(1).split("device")[0].split("dtype")[0])
            if dims:
                analysis["input_shapes"].append([int(d) for d in dims])

        return analysis

    # ----- Starter generation -----

    def generate_starter(self, backend: str = "cuda") -> str:
        """Generate a starter kernel.py (ModelNew initially copies Model logic)."""
        analysis = self.analyze()
        ops_str = ", ".join(analysis["operations"]) or "unknown"

        header = f'''"""
KernelBench Problem {self.uid}: {self.name}
Level: {self.level} | Problem ID: {self.problem_id}
Operations: {ops_str}
Difficulty: {analysis["estimated_difficulty"]}

Source: ScalingIntelligence/KernelBench
Optimized with AutoKernel (https://github.com/RightNow-AI/autokernel)

The agent optimizes ModelNew to outperform the PyTorch reference (Model).
Edit ModelNew.forward() -- use CUDA C++ via compile_cuda() or Triton @jit.
Run `uv run kernelbench/bench_kb.py` to evaluate correctness + speedup.
"""

KERNELBENCH_PROBLEM = {{
    "level": {self.level},
    "problem_id": {self.problem_id},
    "name": {self.name!r},
}}

import torch
import torch.nn as nn
import torch.nn.functional as F
'''

        # Deduplicate imports already in header
        skip_imports = {
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "from torch import nn",
        }
        filtered_lines = []
        for line in self.source_code.split("\n"):
            if line.strip() in skip_imports:
                continue
            # Also skip `from torch.nn import functional as F` and similar
            if re.match(r"^\s*import\s+torch\.nn\.functional\s+as\s+F\s*$", line.strip()):
                continue
            filtered_lines.append(line)
        remaining_source = "\n".join(filtered_lines).strip()

        # Build ModelNew by copying Model class
        model_new_source = self._extract_and_rename_model()

        compile_hint = ""
        if backend == "cuda":
            compile_hint = """
# Optional: use AutoKernel's CUDA compilation utility for custom CUDA C++ kernels
# from kernels.cuda._compile import compile_cuda
#
# CUDA_SRC = r\"""
# #include <torch/extension.h>
# #include <cuda_runtime.h>
# #include <cuda_fp16.h>
#
# __global__ void my_kernel(const float* input, float* output, int N) {
#     int idx = blockIdx.x * blockDim.x + threadIdx.x;
#     if (idx < N) output[idx] = input[idx];
# }
#
# torch::Tensor my_op_cuda(torch::Tensor input) {
#     auto output = torch::empty_like(input);
#     int N = input.numel();
#     my_kernel<<<(N+255)/256, 256>>>(input.data_ptr<float>(), output.data_ptr<float>(), N);
#     return output;
# }
# \"""
# _mod = None
# def _get_mod():
#     global _mod
#     if _mod is None:
#         _mod = compile_cuda(CUDA_SRC, "my_op_cuda")
#     return _mod
"""

        return f"""{header}
{compile_hint}
# ============================================================================
# Reference implementation (DO NOT MODIFY below this line)
# ============================================================================

{remaining_source}

# ============================================================================
# Optimized implementation (EDIT THIS)
# ============================================================================

# ModelNew must produce outputs matching Model within atol=1e-2, rtol=1e-2.
# Start by copying Model's logic, then optimize with CUDA C++ or Triton.

{model_new_source}
"""

    def _extract_and_rename_model(self) -> str:
        """Extract Model class source and produce a renamed ModelNew copy."""
        model_src = self._extract_class("Model")
        if model_src:
            return model_src.replace("class Model(", "class ModelNew(", 1)

        # Fallback: delegate wrapper
        return '''class ModelNew(nn.Module):
    """Optimized version -- replace forward() internals with custom kernels."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._ref = Model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        # TODO: Replace with optimized implementation
        return self._ref(*args, **kwargs)
'''

    def _extract_class(self, class_name: str) -> Optional[str]:
        """Extract a class definition (including body) from source."""
        lines = self.source_code.split("\n")
        collecting = False
        class_lines: List[str] = []

        for line in lines:
            if re.match(rf"^class\s+{class_name}\s*\(", line):
                collecting = True
                class_lines.append(line)
                continue
            if collecting:
                # End of class: non-empty, non-indented, top-level construct
                if line.strip() and not line[0].isspace() and not line.strip().startswith("#"):
                    if re.match(r"^(class |def |@)", line):
                        break
                class_lines.append(line)

        if class_lines:
            # Trim trailing blank lines
            while class_lines and not class_lines[-1].strip():
                class_lines.pop()
            return "\n".join(class_lines)
        return None


# ---------------------------------------------------------------------------
# Problem loading from various sources
# ---------------------------------------------------------------------------

def load_from_huggingface(
    level: Optional[int] = None,
    problem_id: Optional[int] = None,
) -> List[KernelBenchProblem]:
    """Load problems from HuggingFace ScalingIntelligence/KernelBench."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library required for HuggingFace loading.")
        print("       Install: uv pip install datasets")
        sys.exit(1)

    print("Loading KernelBench dataset from HuggingFace...")
    try:
        ds = load_dataset("ScalingIntelligence/KernelBench", split="test")
    except Exception:
        # Try without split (some versions use different splits)
        try:
            ds = load_dataset("ScalingIntelligence/KernelBench")
            # Try common split names
            for split in ["test", "train", "validation"]:
                if split in ds:
                    ds = ds[split]
                    break
            else:
                # Use first available split
                ds = next(iter(ds.values()))
        except Exception as e:
            print(f"ERROR: Failed to load dataset: {e}")
            print("       Check your network connection and HuggingFace access.")
            sys.exit(1)

    problems = []
    for entry in ds:
        p_level = int(entry.get("level", entry.get("Level", 0)))
        p_id = int(entry.get("problem_id", entry.get("Problem_ID", 0)))
        p_name = str(entry.get("name", entry.get("Name", f"problem_{p_id}")))
        p_code = str(entry.get("code", entry.get("Code", "")))

        if not p_code.strip():
            continue
        if level is not None and p_level != level:
            continue
        if problem_id is not None and p_id != problem_id:
            continue

        prob = KernelBenchProblem(
            level=p_level, problem_id=p_id, name=p_name, source_code=p_code,
        )
        prob.save_to_cache()
        problems.append(prob)

    print(f"  Cached {len(problems)} problem(s).")
    return problems


def load_from_local_repo(
    repo_path: str,
    level: Optional[int] = None,
    problem_id: Optional[int] = None,
) -> List[KernelBenchProblem]:
    """Load problems from a local KernelBench git clone."""
    repo = Path(repo_path)
    # KernelBench repo: root/KernelBench/level{N}/*.py  or  root/level{N}/*.py
    kb_dir = repo / "KernelBench" if (repo / "KernelBench").exists() else repo

    problems = []
    levels = range(1, 5) if level is None else [level]

    for lvl in levels:
        level_dir = kb_dir / f"level{lvl}"
        if not level_dir.exists():
            continue
        for py_file in sorted(level_dir.glob("*.py")):
            match = re.match(r"(\d+)_(.+)\.py", py_file.name)
            if not match:
                continue
            p_id = int(match.group(1))
            p_name = match.group(2).replace("_", " ").strip()
            if problem_id is not None and p_id != problem_id:
                continue
            source = py_file.read_text(encoding="utf-8")
            prob = KernelBenchProblem(
                level=lvl, problem_id=p_id, name=p_name, source_code=source,
            )
            prob.save_to_cache()
            problems.append(prob)

    print(f"  Cached {len(problems)} problem(s) from {kb_dir}.")
    return problems


def load_from_file(
    path: str,
    level: int = 1,
    problem_id: int = 0,
) -> KernelBenchProblem:
    """Load a single problem from a standalone .py file."""
    p = Path(path)
    if not p.exists():
        print(f"ERROR: File not found: {path}")
        sys.exit(1)
    source = p.read_text(encoding="utf-8")
    name = p.stem.replace("_", " ")
    prob = KernelBenchProblem(
        level=level, problem_id=problem_id, name=name, source_code=source,
    )
    prob.save_to_cache()
    return prob


# ---------------------------------------------------------------------------
# Cache queries
# ---------------------------------------------------------------------------

def list_cached(level: Optional[int] = None) -> List[Dict[str, Any]]:
    """List all cached problems with metadata."""
    results = []
    levels = range(1, 5) if level is None else [level]
    for lvl in levels:
        level_dir = KB_CACHE_DIR / f"level{lvl}"
        if not level_dir.exists():
            continue
        for meta_file in sorted(level_dir.glob("*.json")):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                results.append(meta)
            except json.JSONDecodeError:
                continue
    return results


def get_problem(level: int, problem_id: int) -> Optional[KernelBenchProblem]:
    """Retrieve a problem from cache. Returns None if not cached."""
    return KernelBenchProblem.load_from_cache(level, problem_id)


# ---------------------------------------------------------------------------
# Workspace setup
# ---------------------------------------------------------------------------

def setup_problem(problem: KernelBenchProblem, backend: str = "cuda") -> None:
    """
    Set up workspace for optimizing a KernelBench problem.

    Creates:
      workspace/kb_active/reference.py   -- original Model + get_inputs
      workspace/kb_active/metadata.json  -- problem metadata + analysis
      kernel.py                          -- starter ModelNew (edit this)
    """
    KB_ACTIVE_DIR.mkdir(parents=True, exist_ok=True)

    # Write reference
    ref_path = KB_ACTIVE_DIR / "reference.py"
    ref_path.write_text(problem.source_code, encoding="utf-8")

    # Write metadata
    meta_path = KB_ACTIVE_DIR / "metadata.json"
    analysis = problem.analyze()
    metadata = {
        "level": problem.level,
        "problem_id": problem.problem_id,
        "name": problem.name,
        "uid": problem.uid,
        "analysis": analysis,
        "backend": backend,
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Generate starter kernel.py
    starter = problem.generate_starter(backend=backend)
    KERNEL_PY.write_text(starter, encoding="utf-8")

    # Report
    print(f"=== KernelBench Problem Setup ===")
    print()
    print(f"  Problem:    {problem.uid} -- {problem.name}")
    print(f"  Level:      {problem.level}")
    print(f"  Operations: {', '.join(analysis['operations']) or 'unknown'}")
    print(f"  Difficulty: {analysis['estimated_difficulty']}")
    print(f"  Parameters: {'yes' if analysis['has_parameters'] else 'no'}")
    print(f"  Forward:    {analysis['forward_lines']} lines")
    if analysis["input_shapes"]:
        for i, s in enumerate(analysis["input_shapes"]):
            print(f"  Input {i}:    shape={s}")
    print()
    print(f"  Reference:  workspace/kb_active/reference.py")
    print(f"  Metadata:   workspace/kb_active/metadata.json")
    print(f"  Kernel:     kernel.py  <-- EDIT THIS")
    print()
    print("Next steps:")
    print("  1. Edit kernel.py -- optimize ModelNew.forward()")
    print("  2. Run: uv run kernelbench/bench_kb.py")
    print("  3. Keep improvements or git reset --hard HEAD~1 to revert")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KernelBench Bridge -- Load and manage KernelBench problems",
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # -- fetch --
    fetch_p = sub.add_parser("fetch", help="Download problems into local cache")
    fetch_p.add_argument("--source", choices=["hf", "local", "file"], default="hf")
    fetch_p.add_argument("--level", type=int, default=None)
    fetch_p.add_argument("--problem", type=int, default=None)
    fetch_p.add_argument("--repo-path", type=str, default=None)
    fetch_p.add_argument("--file-path", type=str, default=None)

    # -- list --
    list_p = sub.add_parser("list", help="List cached problems")
    list_p.add_argument("--level", type=int, default=None)

    # -- info --
    info_p = sub.add_parser("info", help="Show detailed problem info")
    info_p.add_argument("--level", type=int, required=True)
    info_p.add_argument("--problem", type=int, required=True)

    # -- setup --
    setup_p = sub.add_parser("setup", help="Set up workspace for a problem")
    setup_p.add_argument("--level", type=int, required=True)
    setup_p.add_argument("--problem", type=int, required=True)
    setup_p.add_argument("--backend", choices=["cuda", "triton"], default="cuda")
    setup_p.add_argument(
        "--source", choices=["hf", "local", "file"], default=None,
        help="Auto-fetch from this source if problem not in cache",
    )
    setup_p.add_argument("--repo-path", type=str, default=None)
    setup_p.add_argument("--file-path", type=str, default=None)

    args = parser.parse_args()

    if args.command == "fetch":
        if args.source == "hf":
            load_from_huggingface(level=args.level, problem_id=args.problem)
        elif args.source == "local":
            if not args.repo_path:
                print("ERROR: --repo-path required for local source")
                sys.exit(1)
            load_from_local_repo(args.repo_path, level=args.level, problem_id=args.problem)
        elif args.source == "file":
            if not args.file_path:
                print("ERROR: --file-path required for file source")
                sys.exit(1)
            load_from_file(args.file_path, level=args.level or 1, problem_id=args.problem or 0)

    elif args.command == "list":
        cached = list_cached(level=args.level)
        if not cached:
            print("No cached problems. Run 'fetch' first:")
            print("  uv run kernelbench/bridge.py fetch --source hf --level 1")
            return
        print(f"{'Level':<7} {'ID':<6} {'Name'}")
        print("-" * 65)
        for p in cached:
            print(f"{p['level']:<7} {p['problem_id']:<6} {p.get('name', '?')}")
        print(f"\n{len(cached)} problem(s) cached.")

    elif args.command == "info":
        prob = get_problem(args.level, args.problem)
        if prob is None:
            print(f"Problem L{args.level}_P{args.problem:03d} not cached.")
            print("  Fetch first: uv run kernelbench/bridge.py fetch --source hf "
                  f"--level {args.level} --problem {args.problem}")
            sys.exit(1)
        analysis = prob.analyze()
        print(f"Problem {prob.uid}: {prob.name}")
        print(f"  Level:      {prob.level}")
        print(f"  Operations: {', '.join(analysis['operations']) or 'unknown'}")
        print(f"  Difficulty: {analysis['estimated_difficulty']}")
        print(f"  Parameters: {'yes' if analysis['has_parameters'] else 'no'}")
        print(f"  Forward:    {analysis['forward_lines']} lines")
        if analysis["input_shapes"]:
            for i, s in enumerate(analysis["input_shapes"]):
                print(f"  Input {i}:    {s}")
        print(f"\nSource preview ({len(prob.source_code.splitlines())} lines):")
        for i, line in enumerate(prob.source_code.splitlines()[:30], 1):
            print(f"  {i:3d} | {line}")
        total = len(prob.source_code.splitlines())
        if total > 30:
            print(f"  ... ({total} lines total)")

    elif args.command == "setup":
        prob = get_problem(args.level, args.problem)
        if prob is None and args.source:
            # Auto-fetch
            if args.source == "hf":
                probs = load_from_huggingface(level=args.level, problem_id=args.problem)
                prob = probs[0] if probs else None
            elif args.source == "local" and args.repo_path:
                probs = load_from_local_repo(
                    args.repo_path, level=args.level, problem_id=args.problem,
                )
                prob = probs[0] if probs else None
            elif args.source == "file" and args.file_path:
                prob = load_from_file(
                    args.file_path, level=args.level, problem_id=args.problem,
                )
        if prob is None:
            print(f"Problem L{args.level}_P{args.problem:03d} not found.")
            print("  Fetch first: uv run kernelbench/bridge.py fetch --source hf "
                  f"--level {args.level} --problem {args.problem}")
            print("  Or add --source hf to auto-fetch.")
            sys.exit(1)
        setup_problem(prob, backend=args.backend)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  uv run kernelbench/bridge.py fetch --source hf --level 1")
        print("  uv run kernelbench/bridge.py list --level 1")
        print("  uv run kernelbench/bridge.py setup --level 1 --problem 1 --source hf")


if __name__ == "__main__":
    main()
