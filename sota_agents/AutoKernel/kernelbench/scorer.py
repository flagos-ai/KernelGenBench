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
KernelBench Scorer -- Batch evaluation and fast_p metric computation.

Evaluates multiple KernelBench problems, computes the fast_p metric at various
speedup thresholds, and generates a leaderboard-style report.

fast_p = (# problems correct AND speedup >= p) / (total problems)

Usage:
    uv run kernelbench/scorer.py --level 1                  # Score all Level 1 problems
    uv run kernelbench/scorer.py --level 1 --quick           # Quick mode
    uv run kernelbench/scorer.py --level 1 --problems 1-10   # Score problems 1 through 10
    uv run kernelbench/scorer.py --report                     # Show aggregate results
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PROJECT_DIR / "workspace"
KB_CACHE_DIR = WORKSPACE_DIR / "kb_cache"
KB_SCORES_PATH = WORKSPACE_DIR / "kb_scores.json"

# fast_p thresholds
FAST_P_THRESHOLDS = [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0]


def load_scores() -> Dict[str, Any]:
    """Load accumulated scores from disk."""
    if KB_SCORES_PATH.exists():
        try:
            return json.loads(KB_SCORES_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"problems": {}, "metadata": {}}


def save_scores(scores: Dict[str, Any]) -> None:
    """Save scores to disk."""
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    KB_SCORES_PATH.write_text(json.dumps(scores, indent=2), encoding="utf-8")


def compute_fast_p(
    results: List[Dict[str, Any]],
    threshold: float,
) -> float:
    """
    Compute fast_p metric.

    fast_p = (# correct AND speedup >= threshold) / total
    """
    if not results:
        return 0.0
    passing = sum(
        1 for r in results
        if r.get("correctness") == "PASS" and r.get("speedup", 0) >= threshold
    )
    return passing / len(results)


def compute_all_fast_p(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute fast_p at all standard thresholds."""
    return {
        f"fast_{t}": compute_fast_p(results, t)
        for t in FAST_P_THRESHOLDS
    }


def parse_problem_range(range_str: str) -> List[int]:
    """Parse '1-10' or '1,3,5,7' into a list of problem IDs."""
    ids = []
    for part in range_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return ids


def run_single_problem(
    level: int,
    problem_id: int,
    quick: bool = False,
    backend: str = "cuda",
) -> Dict[str, Any]:
    """
    Set up and evaluate a single KernelBench problem.

    Returns a result dict with correctness, speedup, etc.
    """
    import subprocess

    result: Dict[str, Any] = {
        "level": level,
        "problem_id": problem_id,
        "correctness": "FAIL",
        "speedup": 0.0,
        "kernel_time_ms": 0.0,
        "reference_time_ms": 0.0,
        "error": None,
    }

    # Set up the problem
    setup_cmd = [
        sys.executable, str(SCRIPT_DIR / "bridge.py"), "setup",
        "--level", str(level),
        "--problem", str(problem_id),
        "--backend", backend,
        "--source", "hf",
    ]
    try:
        proc = subprocess.run(
            setup_cmd, capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_DIR),
        )
        if proc.returncode != 0:
            result["error"] = f"setup failed: {proc.stderr[:200]}"
            return result
    except subprocess.TimeoutExpired:
        result["error"] = "setup timed out"
        return result
    except Exception as e:
        result["error"] = f"setup error: {e}"
        return result

    # Run bench_kb.py
    bench_cmd = [
        sys.executable, str(SCRIPT_DIR / "bench_kb.py"),
        "--skip-stability", "--skip-determinism",
    ]
    if quick:
        bench_cmd.append("--quick")

    try:
        proc = subprocess.run(
            bench_cmd, capture_output=True, text=True, timeout=300,
            cwd=str(PROJECT_DIR),
        )
        output = proc.stdout + proc.stderr

        # Parse greppable output
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("correctness:"):
                result["correctness"] = line.split(":", 1)[1].strip()
            elif line.startswith("speedup:"):
                val = line.split(":", 1)[1].strip().replace("x", "")
                try:
                    result["speedup"] = float(val)
                except ValueError:
                    pass
            elif line.startswith("kernel_time_ms:"):
                try:
                    result["kernel_time_ms"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.startswith("reference_time_ms:"):
                try:
                    result["reference_time_ms"] = float(line.split(":", 1)[1].strip())
                except ValueError:
                    pass

    except subprocess.TimeoutExpired:
        result["error"] = "benchmark timed out"
    except Exception as e:
        result["error"] = f"benchmark error: {e}"

    return result


def run_level(
    level: int,
    problem_ids: Optional[List[int]] = None,
    quick: bool = False,
    backend: str = "cuda",
) -> List[Dict[str, Any]]:
    """Run all (or selected) problems in a level."""

    # Discover available problems
    level_dir = KB_CACHE_DIR / f"level{level}"
    available_ids = []
    if level_dir.exists():
        for f in sorted(level_dir.glob("*.json")):
            try:
                meta = json.loads(f.read_text(encoding="utf-8"))
                available_ids.append(meta["problem_id"])
            except (json.JSONDecodeError, KeyError):
                continue

    if not available_ids:
        print(f"No cached problems for Level {level}.")
        print(f"  Fetch first: uv run kernelbench/bridge.py fetch --source hf --level {level}")
        return []

    # Filter if specific IDs requested
    if problem_ids:
        run_ids = [pid for pid in problem_ids if pid in available_ids]
        if not run_ids:
            print(f"None of the requested problem IDs found in cache for Level {level}.")
            return []
    else:
        run_ids = available_ids

    print(f"=== KernelBench Scorer: Level {level} ({len(run_ids)} problems) ===\n")

    results = []
    scores = load_scores()

    for i, pid in enumerate(run_ids):
        print(f"[{i + 1}/{len(run_ids)}] Problem L{level}_P{pid:03d}...", end=" ", flush=True)
        t0 = time.time()

        result = run_single_problem(level, pid, quick=quick, backend=backend)
        elapsed = time.time() - t0

        result["elapsed_s"] = elapsed
        results.append(result)

        # Save incrementally
        key = f"L{level}_P{pid:03d}"
        scores["problems"][key] = result
        save_scores(scores)

        # Print one-line result
        status = result["correctness"]
        speedup = result["speedup"]
        if result.get("error"):
            print(f"ERROR ({result['error'][:60]})")
        else:
            print(f"{status} | speedup={speedup:.2f}x | {elapsed:.1f}s")

    return results


def print_report(results: Optional[List[Dict]] = None) -> None:
    """Print a leaderboard-style report."""
    scores = load_scores()

    if results is None:
        results = list(scores.get("problems", {}).values())

    if not results:
        print("No results found. Run scorer first.")
        return

    # Group by level
    by_level: Dict[int, List[Dict]] = {}
    for r in results:
        lvl = r.get("level", 0)
        by_level.setdefault(lvl, []).append(r)

    print("\n" + "=" * 70)
    print("KERNELBENCH RESULTS")
    print("=" * 70)

    for lvl in sorted(by_level):
        level_results = by_level[lvl]
        n_total = len(level_results)
        n_correct = sum(1 for r in level_results if r.get("correctness") == "PASS")
        n_errors = sum(1 for r in level_results if r.get("error"))

        fast_p = compute_all_fast_p(level_results)

        print(f"\nLevel {lvl}: {n_total} problems")
        print(f"  Correct:    {n_correct}/{n_total} ({100 * n_correct / n_total:.1f}%)")
        if n_errors:
            print(f"  Errors:     {n_errors}")

        print(f"  fast_p scores:")
        for key, val in fast_p.items():
            bar = "#" * int(val * 40) + "." * (40 - int(val * 40))
            print(f"    {key:>8}: {val:.3f} [{bar}]")

        # Top speedups
        correct_results = [r for r in level_results if r.get("correctness") == "PASS"]
        if correct_results:
            top = sorted(correct_results, key=lambda r: r.get("speedup", 0), reverse=True)[:5]
            print(f"  Top speedups:")
            for r in top:
                pid = r.get("problem_id", "?")
                print(f"    P{pid:03d}: {r.get('speedup', 0):.2f}x")

    # Aggregate across all levels
    all_results = [r for rs in by_level.values() for r in rs]
    if len(by_level) > 1:
        print(f"\nAggregate ({len(all_results)} problems):")
        agg_fast_p = compute_all_fast_p(all_results)
        for key, val in agg_fast_p.items():
            print(f"  {key:>8}: {val:.3f}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="KernelBench Scorer -- Batch evaluation and fast_p computation",
    )
    parser.add_argument("--level", type=int, default=None,
                        help="Level to evaluate (1-4)")
    parser.add_argument("--problems", type=str, default=None,
                        help="Problem range: '1-10' or '1,3,5' (default: all)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode for each problem")
    parser.add_argument("--backend", choices=["cuda", "triton"], default="cuda",
                        help="Backend for starter kernels")
    parser.add_argument("--report", action="store_true",
                        help="Just print report from existing results")

    args = parser.parse_args()

    if args.report:
        print_report()
        return

    if args.level is None:
        print("ERROR: --level required (1-4)")
        parser.print_help()
        sys.exit(1)

    problem_ids = None
    if args.problems:
        problem_ids = parse_problem_range(args.problems)

    results = run_level(
        args.level,
        problem_ids=problem_ids,
        quick=args.quick,
        backend=args.backend,
    )

    if results:
        print_report(results)


if __name__ == "__main__":
    main()
