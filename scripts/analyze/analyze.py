#!/usr/bin/env python3
"""
Analyze speedup data from verification results (v6).

Key features in v6:
- Auto-detect dataset (v2_1=110 ops, KernelGenBench=210 ops)
- Per-type breakdown (aten / cublas / vllm13)
- Robust statistics: geometric mean, median, IQM (interquartile mean)
- Per-operator stats include median and geometric mean
- ONLY uses verification/log_*/result.json
- Keeps BEST speedup across all rounds

Usage:
    python scripts/analyze/analyze.py <result_dir> [--dataset auto|v2_1|KernelGenBench] [--no-antihack]
"""

import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


# ── Dataset definitions ──────────────────────────────────────────────────────

DATASET_TOTALS = {
    "v2_1": {"total": 110, "aten": 110, "cublas": 0, "vllm13": 0},
    "KernelGenBench": {"total": 210, "aten": 110, "cublas": 50, "vllm13": 50},
}


def detect_dataset(operator_results: Dict[str, Dict]) -> str:
    """Auto-detect dataset from operator name prefixes."""
    has_cublas = any(k.startswith("cublas::") for k in operator_results)
    has_vllm = any(k.startswith("vllm13::") for k in operator_results)
    if has_cublas or has_vllm:
        return "KernelGenBench"
    return "v2_1"


def get_op_type(op_name: str) -> str:
    """Classify operator by prefix."""
    if op_name.startswith("cublas::"):
        return "cublas"
    if op_name.startswith("vllm13::") or op_name.startswith("vllm15::"):
        return "vllm13"
    return "aten"


# ── Robust statistics ────────────────────────────────────────────────────────

def geometric_mean(values: List[float]) -> Optional[float]:
    """Compute geometric mean. Handles zeros by filtering them out."""
    positive = [v for v in values if v > 0]
    if not positive:
        return None
    log_sum = sum(math.log(v) for v in positive)
    return math.exp(log_sum / len(positive))


def median(values: List[float]) -> Optional[float]:
    """Compute median."""
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


def iqm(values: List[float]) -> Optional[float]:
    """Interquartile Mean: mean of values between Q1 and Q3."""
    if len(values) < 4:
        return geometric_mean(values)  # fallback for small samples
    s = sorted(values)
    n = len(s)
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    middle = s[q1_idx:q3_idx]
    if not middle:
        return None
    return sum(middle) / len(middle)


def geometric_std(values: List[float]) -> Optional[float]:
    """Geometric standard deviation = exp(std(log(values))). Reports spread as multiplier."""
    positive = [v for v in values if v > 0]
    if len(positive) < 2:
        return None
    logs = [math.log(v) for v in positive]
    mean_log = sum(logs) / len(logs)
    variance = sum((x - mean_log) ** 2 for x in logs) / (len(logs) - 1)
    return math.exp(math.sqrt(variance))


def trimmed_mean(values: List[float], pct: float = 0.1) -> Optional[float]:
    """Trimmed mean: remove top/bottom pct fraction, then average."""
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    trim = max(1, int(n * pct))
    if 2 * trim >= n:
        return sum(s) / len(s)
    trimmed = s[trim:n - trim]
    return sum(trimmed) / len(trimmed)


# ── Core logic ───────────────────────────────────────────────────────────────

def compute_speedup_stats(speedup_list: List[Dict]) -> Optional[Dict]:
    """Compute speedup statistics from speedup list with robust metrics."""
    if not speedup_list:
        return None

    speedups = [e["speedup"] for e in speedup_list if "speedup" in e and e["speedup"] is not None]
    ref_times = [e["ref_time"] for e in speedup_list if "ref_time" in e and e["ref_time"] is not None]
    res_times = [e["res_time"] for e in speedup_list if "res_time" in e and e["res_time"] is not None]

    if not speedups:
        return None

    return {
        "avg_speedup": sum(speedups) / len(speedups),
        "geo_mean": geometric_mean(speedups),
        "median": median(speedups),
        "iqm": iqm(speedups),
        "min_speedup": min(speedups),
        "max_speedup": max(speedups),
        "avg_ref_time": sum(ref_times) / len(ref_times) if ref_times else None,
        "avg_res_time": sum(res_times) / len(res_times) if res_times else None,
        "num_tests": len(speedups),
    }


def _load_passed_operators(result_dir: Path) -> Optional[set]:
    """Load passed operator names from pass_at_k results.

    Priority: pass_at_k_results_antihack.json > pass_at_k_results.json
    Returns set of passed operator names, or None if not available.
    """
    # Priority 1: antihack clean list
    antihack_file = result_dir / "pass_at_k_results_antihack.json"
    if antihack_file.exists():
        try:
            with open(antihack_file, 'r') as f:
                data = json.load(f)
            clean = data.get("clean_passed_operators")
            if clean is not None:
                print(f"Loaded {len(clean)} passed operators from antihack results")
                return set(clean)
        except Exception as e:
            print(f"Warning: Failed to load {antihack_file}: {e}", file=sys.stderr)

    # Priority 2: pass_at_k_results.json
    results_file = result_dir / "pass_at_k_results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            passed = data.get("passed_operators")
            if passed is not None:
                print(f"Loaded {len(passed)} passed operators from pass_at_k_results")
                return set(passed)
        except Exception as e:
            print(f"Warning: Failed to load {results_file}: {e}", file=sys.stderr)

    return None


def _load_speedup_from_test_reports(
    log_dir: Path, op_name: str, round_num: int
) -> Optional[Dict]:
    """Load speedup data from individual test_report_*.json file.

    Returns operator result dict with speedup_stats, or None.
    """
    report_file = log_dir / f"test_report_{op_name}.json"
    if not report_file.exists():
        return None

    try:
        with open(report_file, 'r') as f:
            test_cases = json.load(f)
    except Exception:
        return None

    if not test_cases or not isinstance(test_cases, list):
        return None

    # Collect speedup entries from individual test cases
    speedup_list = []
    for tc in test_cases:
        sp = tc.get("speedup")
        if sp and isinstance(sp, dict) and "speedup" in sp:
            speedup_list.append(sp)

    speedup_stats = compute_speedup_stats(speedup_list)
    return {
        "round": round_num,
        "speedup_stats": speedup_stats,
    }


def load_from_verification_dir(result_dir: Path) -> Dict[str, Dict]:
    """Load results from verification directory.

    Uses a multi-source strategy:
    1. Get passed operators from pass_at_k results (antihack > results)
    2. Load speedup from result.json where available
    3. For passed operators missing from result.json, load from test_report_*.json

    Keeps the BEST speedup (by geometric mean) across all rounds.
    """
    verification_dir = result_dir / "verification"
    if not verification_dir.exists():
        print(f"Error: verification directory not found: {verification_dir}", file=sys.stderr)
        return {}

    operator_results = {}

    log_dirs = sorted(
        [d for d in verification_dir.iterdir() if d.is_dir() and d.name.startswith("log_")],
        key=lambda x: int(x.name.split("_")[1])
    )

    if not log_dirs:
        print(f"Error: No log_* directories found in {verification_dir}", file=sys.stderr)
        return {}

    print(f"Found {len(log_dirs)} rounds of verification results")

    # Get authoritative passed operator set
    passed_ops = _load_passed_operators(result_dir)

    for log_dir in log_dirs:
        round_num = int(log_dir.name.split("_")[1])
        result_file = log_dir / "result.json"

        # Phase 1: Load from result.json (operators marked success)
        result_json_passed = set()
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    results = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {result_file}: {e}", file=sys.stderr)
                results = []

            for item in results:
                op_name = item.get("op_name")
                if not op_name:
                    continue

                if not item.get("success"):
                    continue

                result_json_passed.add(op_name)
                speedup_list = item.get("speedup", [])
                speedup_stats = compute_speedup_stats(speedup_list)
                current_geo = speedup_stats["geo_mean"] if speedup_stats else None

                if op_name in operator_results:
                    existing_stats = operator_results[op_name].get("speedup_stats")
                    existing_geo = existing_stats["geo_mean"] if existing_stats else None
                    if current_geo is not None:
                        if existing_geo is None or current_geo > existing_geo:
                            operator_results[op_name] = {
                                "round": round_num,
                                "speedup_stats": speedup_stats,
                            }
                else:
                    operator_results[op_name] = {
                        "round": round_num,
                        "speedup_stats": speedup_stats,
                    }

        # Phase 2: For passed operators not in result.json, try test_report files
        if passed_ops is not None:
            missing_ops = passed_ops - result_json_passed - set(operator_results.keys())
            for op_name in missing_ops:
                report_result = _load_speedup_from_test_reports(log_dir, op_name, round_num)
                if report_result is not None:
                    current_geo = report_result["speedup_stats"]["geo_mean"] if report_result["speedup_stats"] else None
                    if op_name in operator_results:
                        existing_stats = operator_results[op_name].get("speedup_stats")
                        existing_geo = existing_stats["geo_mean"] if existing_stats else None
                        if current_geo is not None:
                            if existing_geo is None or current_geo > existing_geo:
                                operator_results[op_name] = report_result
                    else:
                        operator_results[op_name] = report_result

    # Phase 3: If we have a passed_ops set, ensure all passed ops are represented
    # (even without speedup data)
    if passed_ops is not None:
        for op_name in passed_ops:
            if op_name not in operator_results:
                operator_results[op_name] = {
                    "round": 0,
                    "speedup_stats": None,
                }

    return operator_results


def load_antihack_clean_ops(result_dir: Path) -> Optional[set]:
    """Load clean_passed_operators from pass_at_k_results_antihack.json as whitelist.

    Returns a set of clean operator names, or None if not available.
    """
    antihack_file = result_dir / "pass_at_k_results_antihack.json"
    if not antihack_file.exists():
        return None
    try:
        with open(antihack_file, 'r') as f:
            data = json.load(f)
        clean = data.get("clean_passed_operators")
        if clean is not None:
            return set(clean)
        return None
    except Exception as e:
        print(f"Warning: Failed to load antihack clean ops: {e}", file=sys.stderr)
        return None


def load_antihack_hacked_ops(result_dir: Path) -> List[str]:
    """Load hacked operator names from antihack results.

    Searches for pass_at_k_results_antihack.json or antihack_report.json.
    Returns list of hacked operator names to exclude.
    """
    # Try pass_at_k format first
    antihack_file = result_dir / "pass_at_k_results_antihack.json"
    if not antihack_file.exists():
        # Try agent format
        antihack_file = result_dir / "antihack_report.json"
    if not antihack_file.exists():
        # Try antihack_round_*.json (pass@1 format)
        candidates = sorted(result_dir.glob("antihack_round_*.json"))
        if candidates:
            antihack_file = candidates[-1]  # use latest round
    if not antihack_file.exists():
        return []

    try:
        with open(antihack_file, 'r') as f:
            data = json.load(f)
        hacked = data.get("hacked_operators", [])
        # Agent format: list of dicts with 'op_name'
        if hacked and isinstance(hacked[0], dict):
            hacked = [h["op_name"] for h in hacked]
        if hacked:
            print(f"Anti-hack: found {len(hacked)} hacked operators to exclude: {hacked}")
        return hacked
    except Exception as e:
        print(f"Warning: Failed to load antihack results: {e}", file=sys.stderr)
        return []


# ── Grouping helpers ─────────────────────────────────────────────────────────

def group_by_type(operator_results: Dict[str, Dict]) -> Dict[str, Dict[str, Dict]]:
    """Group operators by type (aten/cublas/vllm13)."""
    groups = {"aten": {}, "cublas": {}, "vllm13": {}}
    for op_name, data in operator_results.items():
        t = get_op_type(op_name)
        groups[t][op_name] = data
    return groups


def collect_geo_means(ops: Dict[str, Dict]) -> List[float]:
    """Collect geometric means from operators with speedup data."""
    result = []
    for data in ops.values():
        stats = data.get("speedup_stats")
        if stats and stats.get("geo_mean") is not None:
            result.append(stats["geo_mean"])
    return result


def compute_summary_stats(values: List[float]) -> Dict:
    """Compute all summary statistics for a list of speedup values."""
    if not values:
        return {}
    return {
        "count": len(values),
        "arith_mean": sum(values) / len(values),
        "geo_mean": geometric_mean(values),
        "geo_std": geometric_std(values),
        "median": median(values),
        "iqm": iqm(values),
        "trimmed_mean_10": trimmed_mean(values, 0.1),
        "min": min(values),
        "max": max(values),
    }


# ── Timing data ──────────────────────────────────────────────────────────────

def load_timing_data(result_dir: Path) -> Dict[str, Dict]:
    """Load per-operator timing from workspace file timestamps.

    For new format (attempt files): sums duration across all attempts.
    For old format (overwritten files): uses last attempt only.
    Falls back to progress.json duration_seconds.
    """
    timing = {}

    # Scan workspace files for per-attempt timing (support workspaces and workspaces_ako)
    workspaces_dir = result_dir / "workspaces"
    if not workspaces_dir.exists():
        workspaces_dir = result_dir / "workspaces_ako"
    if workspaces_dir.exists():
        for ws_dir in workspaces_dir.iterdir():
            if not ws_dir.is_dir():
                continue
            op_name = ws_dir.name.replace("__", "::", 1)

            # New format: prompt_attempt*.md + cc_output_attempt*.jsonl
            attempt_prompts = sorted(ws_dir.glob("prompt_attempt*.md"))
            attempt_outputs = sorted(ws_dir.glob("cc_output_attempt*.jsonl"))

            if attempt_prompts and attempt_outputs:
                total_duration = 0
                for pf, of in zip(attempt_prompts, attempt_outputs):
                    total_duration += max(0, of.stat().st_mtime - pf.stat().st_mtime)
                timing[op_name] = {
                    "total_duration": round(total_duration),
                    "num_attempts": len(attempt_prompts),
                }
            else:
                # Old format: single prompt.md + cc_output.jsonl
                pf = ws_dir / "prompt.md"
                of = ws_dir / "cc_output.jsonl"
                if pf.exists() and of.exists():
                    timing[op_name] = {
                        "total_duration": round(max(0, of.stat().st_mtime - pf.stat().st_mtime)),
                        "num_attempts": 1,
                    }

    # Fill from progress.json for any missing
    progress_file = result_dir / "progress.json"
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            # Support both "operators" and "kernels" keys
            progress_ops = progress_data.get("operators", {}) or progress_data.get("kernels", {})
            for raw_name, info in progress_ops.items():
                op_name = raw_name.replace("__", "::", 1)
                if op_name not in timing and info.get("duration_seconds"):
                    timing[op_name] = {
                        "total_duration": info["duration_seconds"],
                        "num_attempts": info.get("attempt", 0) + 1,
                    }
        except Exception:
            pass

    return timing


def format_duration(seconds: Optional[int]) -> str:
    """Format seconds as hours (e.g., 1.23h, 0.05h)."""
    if seconds is None:
        return "N/A"
    hours = int(seconds) / 3600
    return f"{hours:.2f}h"


# ── Console output ───────────────────────────────────────────────────────────

def print_summary(operator_results: Dict[str, Dict], result_dir: Path, dataset: str, timing: Dict[str, Dict] = None, token_info: Dict = None):
    """Print summary statistics to console."""
    totals = DATASET_TOTALS[dataset]
    groups = group_by_type(operator_results)

    with_speedup = {k: v for k, v in operator_results.items() if v.get("speedup_stats")}
    without_speedup = {k: v for k, v in operator_results.items() if not v.get("speedup_stats")}

    print()
    print("=" * 110)
    print(f"Speedup Analysis (v6) for: {result_dir.name}")
    print(f"Dataset: {dataset} ({totals['total']} operators)")
    print("=" * 110)
    print()
    print(f"Total passed operators: {len(operator_results)} / {totals['total']}")
    print(f"  - With speedup data: {len(with_speedup)}")
    print(f"  - Without speedup data: {len(without_speedup)}")
    print()

    if not with_speedup:
        print("No speedup data available.")
        return

    # Per-type summary
    all_geo_means = collect_geo_means(with_speedup)
    overall = compute_summary_stats(all_geo_means)

    print("-" * 110)
    print(f"{'Category':<15} {'Passed':>8} {'Total':>8} {'GeoMean':>10} {'GeoStd':>8} {'Median':>10} {'IQM':>10} {'ArithMean':>10} {'Min':>10} {'Max':>10}")
    print("-" * 118)

    for label, type_key in [("Overall", None), ("aten", "aten"), ("cublas", "cublas"), ("vllm13", "vllm13")]:
        if type_key is None:
            ops = with_speedup
            total = totals["total"]
        else:
            ops = {k: v for k, v in groups[type_key].items() if v.get("speedup_stats")}
            total = totals.get(type_key, 0)
            if total == 0:
                continue

        geos = collect_geo_means(ops)
        s = compute_summary_stats(geos)
        if not s:
            print(f"{label:<15} {0:>8} {total:>8} {'N/A':>10} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue

        geo_std_str = f"{s['geo_std']:.4f}" if s.get('geo_std') else "N/A"
        print(f"{label:<15} {s['count']:>8} {total:>8} {s['geo_mean']:>10.4f} {geo_std_str:>8} {s['median']:>10.4f} {s['iqm']:>10.4f} {s['arith_mean']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}")

    print("-" * 118)
    print()

    # Detailed per-operator table
    print("-" * 110)
    print(f"{'Operator':<45} {'Rnd':>4} {'GeoMean':>9} {'Median':>9} {'IQM':>9} {'Mean':>9} {'Min':>9} {'Max':>9} {'N':>6}")
    print("-" * 110)

    sorted_ops = sorted(
        with_speedup.items(),
        key=lambda x: x[1]["speedup_stats"]["geo_mean"] if x[1]["speedup_stats"] and x[1]["speedup_stats"]["geo_mean"] else 0,
        reverse=True
    )

    for op_name, data in sorted_ops:
        stats = data["speedup_stats"]
        if not stats:
            continue
        geo = stats["geo_mean"] or 0
        med = stats["median"] or 0
        iq = stats["iqm"] or 0
        avg = stats["avg_speedup"]
        mn = stats["min_speedup"]
        mx = stats["max_speedup"]
        n = stats["num_tests"]
        rnd = data.get("round", "?")
        print(f"{op_name:<45} {rnd:>4} {geo:>9.4f} {med:>9.4f} {iq:>9.4f} {avg:>9.4f} {mn:>9.4f} {mx:>9.4f} {n:>6}")

    print("-" * 110)
    print()

    # Distribution
    print("=" * 80)
    print("Speedup Distribution (by per-operator geometric mean)")
    print("=" * 80)
    ranges = [
        ("> 2.0x",            lambda x: x > 2.0),
        ("1.5x ~ 2.0x",       lambda x: 1.5 < x <= 2.0),
        ("1.0x ~ 1.5x",       lambda x: 1.0 < x <= 1.5),
        ("0.8x ~ 1.0x",       lambda x: 0.8 < x <= 1.0),
        ("0.5x ~ 0.8x",       lambda x: 0.5 < x <= 0.8),
        ("<= 0.5x (slow)",    lambda x: x <= 0.5),
    ]

    for label, condition in ranges:
        count = sum(1 for s in all_geo_means if condition(s))
        pct = count / len(all_geo_means) * 100 if all_geo_means else 0
        bar = "#" * int(pct / 2)
        print(f"  {label:<20} {count:>3} ({pct:>5.1f}%) {bar}")
    print()

    # Geomean threshold counts (based on total operators tested, not just passed)
    thresholds = [(">0.8", 0.8), (">1.0", 1.0), (">1.5", 1.5)]
    print("Geomean Threshold Counts (based on total tested):")
    for label, thresh in thresholds:
        cnt = sum(1 for g in all_geo_means if g > thresh)
        pct = cnt / totals['total'] * 100 if totals['total'] > 0 else 0
        print(f"  {label:<8} {cnt:>3} / {totals['total']} ({pct:.1f}%)")
    print()

    # Per-type breakdown: thresholds, tokens, timing
    print("=" * 130)
    print("Per-Type Breakdown")
    print("=" * 130)

    type_entries = [("Overall", None), ("aten", "aten"), ("cublas", "cublas"), ("vllm13", "vllm13")]
    for label, type_key in type_entries:
        if type_key is None:
            type_ops = with_speedup
            total = totals["total"]
        else:
            type_ops = {k: v for k, v in groups[type_key].items() if v.get("speedup_stats")}
            total = totals.get(type_key, 0)
            if total == 0:
                continue

        type_geos = collect_geo_means(type_ops)
        if not type_geos:
            continue

        print(f"\n--- {label} ({len(type_geos)} passed / {total} total) ---")

        # Thresholds
        for thresh_label, thresh in thresholds:
            cnt = sum(1 for g in type_geos if g > thresh)
            pct = cnt / total * 100 if total > 0 else 0
            print(f"  {thresh_label:<8} {cnt:>3} / {total} ({pct:.1f}%)")

        # Token stats for this type
        if token_info:
            per_op_tokens = token_info.get("per_op", {})
            # Total tokens = all ops in this type (including failed)
            if type_key is None:
                type_token_total = sum(per_op_tokens.values())
            else:
                prefix = type_key + "__"
                type_token_total = sum(v for k, v in per_op_tokens.items() if k.startswith(prefix))
            passed_count = len(type_ops)
            if type_token_total > 0:
                per_success = type_token_total // passed_count if passed_count > 0 else 0
                print(f"  Tokens: {type_token_total:,} total, {per_success:,} per success")

        # Timing stats for this type
        if timing:
            type_time_total = 0
            type_time_count = 0
            for op_name in type_ops:
                if op_name in timing:
                    type_time_total += timing[op_name]["total_duration"]
                    type_time_count += 1
                else:
                    ws_name = op_name.replace("::", "__", 1)
                    if ws_name in timing:
                        type_time_total += timing[ws_name]["total_duration"]
                        type_time_count += 1
            if type_time_count > 0:
                print(f"  Timing (passed): {format_duration(type_time_total)} total, {format_duration(type_time_total // type_time_count)} avg")

    print()

    # Timing summary
    if timing:
        total_time = sum(t["total_duration"] for t in timing.values())
        avg_time = total_time / len(timing) if timing else 0
        print("=" * 80)
        print("Timing Summary")
        print("=" * 80)
        print(f"  Total time (all operators, all attempts): {format_duration(total_time)}")
        print(f"  Average per operator: {format_duration(int(avg_time))}")
        print(f"  Operators with timing data: {len(timing)}")
        print()

        # Per-operator timing table
        print("-" * 80)
        print(f"{'Operator':<45} {'Duration':>10} {'Attempts':>8}")
        print("-" * 80)
        sorted_timing = sorted(timing.items(), key=lambda x: x[1]["total_duration"], reverse=True)
        for op_name, t in sorted_timing:
            print(f"{op_name:<45} {format_duration(t['total_duration']):>10} {t['num_attempts']:>8}")
        print("-" * 80)
        print()


# ── Markdown output ──────────────────────────────────────────────────────────

def generate_markdown(operator_results: Dict[str, Dict], result_dir: Path, dataset: str, timing: Dict[str, Dict] = None) -> str:
    """Generate markdown report."""
    totals = DATASET_TOTALS[dataset]
    groups = group_by_type(operator_results)
    lines = []

    with_speedup = {k: v for k, v in operator_results.items() if v.get("speedup_stats")}
    without_speedup = {k: v for k, v in operator_results.items() if not v.get("speedup_stats")}

    # Header
    lines.append("# Speedup Analysis Report (v6)")
    lines.append("")
    lines.append(f"**Result Directory:** `{result_dir.name}`")
    lines.append(f"**Dataset:** {dataset} ({totals['total']} operators)")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Data Source:** verification/log_*/result.json")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total passed:** {len(operator_results)} / {totals['total']}")
    lines.append(f"- **With speedup data:** {len(with_speedup)}")
    lines.append(f"- **Without speedup data:** {len(without_speedup)}")
    lines.append("")

    if not with_speedup:
        lines.append("No speedup data available.")
        return "\n".join(lines)

    # Per-type statistics table
    lines.append("## Statistics by Category")
    lines.append("")
    lines.append("| Category | Passed | Total | GeoMean | GeoStd | Median | IQM | ArithMean | Min | Max |")
    lines.append("|----------|--------|-------|---------|--------|--------|-----|-----------|-----|-----|")

    for label, type_key in [("**Overall**", None), ("aten", "aten"), ("cublas", "cublas"), ("vllm13", "vllm13")]:
        if type_key is None:
            ops = with_speedup
            total = totals["total"]
        else:
            ops = {k: v for k, v in groups[type_key].items() if v.get("speedup_stats")}
            total = totals.get(type_key, 0)
            if total == 0:
                continue

        geos = collect_geo_means(ops)
        s = compute_summary_stats(geos)
        if not s:
            lines.append(f"| {label} | 0 | {total} | - | - | - | - | - | - | - |")
            continue
        geo_std_str = f"{s['geo_std']:.4f}" if s.get('geo_std') else "-"
        lines.append(f"| {label} | {s['count']} | {total} | {s['geo_mean']:.4f} | {geo_std_str} | {s['median']:.4f} | {s['iqm']:.4f} | {s['arith_mean']:.4f} | {s['min']:.4f} | {s['max']:.4f} |")

    lines.append("")

    # Distribution
    all_geo_means = collect_geo_means(with_speedup)

    lines.append("## Speedup Distribution")
    lines.append("")
    lines.append("Based on per-operator geometric mean speedup.")
    lines.append("")
    lines.append("| Range | Count | Percentage |")
    lines.append("|-------|-------|------------|")

    ranges = [
        ("> 2.0x",            lambda x: x > 2.0),
        ("1.5x ~ 2.0x",       lambda x: 1.5 < x <= 2.0),
        ("1.0x ~ 1.5x",       lambda x: 1.0 < x <= 1.5),
        ("0.8x ~ 1.0x",       lambda x: 0.8 < x <= 1.0),
        ("0.5x ~ 0.8x",       lambda x: 0.5 < x <= 0.8),
        ("<= 0.5x (slow)",    lambda x: x <= 0.5),
    ]

    for label, condition in ranges:
        count = sum(1 for s in all_geo_means if condition(s))
        pct = count / len(all_geo_means) * 100 if all_geo_means else 0
        lines.append(f"| {label} | {count} | {pct:.1f}% |")
    lines.append("")

    # Detailed per-operator table
    lines.append("## Detailed Results")
    lines.append("")
    lines.append("Sorted by geometric mean (descending).")
    lines.append("")
    lines.append("| Operator | Type | Rnd | GeoMean | Median | IQM | Mean | Min | Max | Tests |")
    lines.append("|----------|------|-----|---------|--------|-----|------|-----|-----|-------|")

    sorted_ops = sorted(
        with_speedup.items(),
        key=lambda x: x[1]["speedup_stats"]["geo_mean"] if x[1]["speedup_stats"] and x[1]["speedup_stats"]["geo_mean"] else 0,
        reverse=True
    )

    for op_name, data in sorted_ops:
        stats = data["speedup_stats"]
        if not stats:
            continue
        op_type = get_op_type(op_name)
        geo = stats["geo_mean"] or 0
        med = stats["median"] or 0
        iq = stats["iqm"] or 0
        avg = stats["avg_speedup"]
        mn = stats["min_speedup"]
        mx = stats["max_speedup"]
        n = stats["num_tests"]
        rnd = data.get("round", "?")
        lines.append(f"| {op_name} | {op_type} | {rnd} | {geo:.4f} | {med:.4f} | {iq:.4f} | {avg:.4f} | {mn:.4f} | {mx:.4f} | {n} |")

    lines.append("")

    # Operators without speedup data
    if without_speedup:
        lines.append("## Operators Without Speedup Data")
        lines.append("")
        lines.append("| Operator | Type | Round |")
        lines.append("|----------|------|-------|")
        for op_name, data in sorted(without_speedup.items()):
            round_val = data.get("round", "N/A")
            lines.append(f"| {op_name} | {get_op_type(op_name)} | {round_val} |")
        lines.append("")

    # Geomean threshold counts
    all_geo_means = collect_geo_means(with_speedup)
    if all_geo_means:
        lines.append("## Geomean Threshold Counts")
        lines.append("")
        lines.append("| Threshold | Count | Total | Percentage |")
        lines.append("|-----------|-------|-------|------------|")
        for label, thresh in [(">0.8", 0.8), (">1.0", 1.0), (">1.5", 1.5)]:
            cnt = sum(1 for g in all_geo_means if g > thresh)
            pct = cnt / len(all_geo_means) * 100
            lines.append(f"| {label} | {cnt} | {len(all_geo_means)} | {pct:.1f}% |")
        lines.append("")

    # Timing summary
    if timing:
        total_time = sum(t["total_duration"] for t in timing.values())
        avg_time = total_time / len(timing) if timing else 0
        lines.append("## Timing Summary")
        lines.append("")
        lines.append(f"- Total time (all operators, all attempts): {format_duration(total_time)}")
        lines.append(f"- Average per operator: {format_duration(int(avg_time))}")
        lines.append(f"- Operators with timing data: {len(timing)}")
        lines.append("")
        lines.append("| Operator | Duration | Attempts |")
        lines.append("|----------|----------|----------|")
        sorted_timing = sorted(timing.items(), key=lambda x: x[1]["total_duration"], reverse=True)
        for op_name, t in sorted_timing:
            lines.append(f"| {op_name} | {format_duration(t['total_duration'])} | {t['num_attempts']} |")
        lines.append("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def save_results(operator_results: Dict[str, Dict], result_dir: Path, dataset: str, timing: Dict[str, Dict] = None):
    """Save results to markdown file."""
    md_content = generate_markdown(operator_results, result_dir, dataset, timing)
    output_file = result_dir / "speedup_analysis_v6.md"
    with open(output_file, 'w') as f:
        f.write(md_content)
    print(f"Results saved to: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze/analyze_speedup_v6.py <result_dir> [--dataset auto|v2_1|KernelGenBench] [--no-antihack]")
        sys.exit(1)

    result_dir = Path(sys.argv[1])
    if not result_dir.exists():
        print(f"Error: Directory does not exist: {result_dir}", file=sys.stderr)
        sys.exit(1)

    # Parse flags
    dataset_override = None
    apply_antihack = True
    for i, arg in enumerate(sys.argv):
        if arg == "--dataset" and i + 1 < len(sys.argv):
            dataset_override = sys.argv[i + 1]
        if arg == "--no-antihack":
            apply_antihack = False

    # Load results
    operator_results = load_from_verification_dir(result_dir)
    if not operator_results:
        print("No successful operators found.", file=sys.stderr)
        sys.exit(1)

    # Apply antihack filtering (default: on)
    if apply_antihack:
        clean_ops = load_antihack_clean_ops(result_dir)
        if clean_ops is not None:
            # Whitelist mode: only keep operators in clean_passed_operators
            operator_results = {k: v for k, v in operator_results.items() if k in clean_ops}
            print(f"Anti-hack (whitelist): {len(operator_results)} clean operators remaining")
        else:
            # Fallback: blacklist mode using hacked_operators
            hacked_ops = load_antihack_hacked_ops(result_dir)
            if hacked_ops:
                for op in hacked_ops:
                    if op in operator_results:
                        del operator_results[op]
                print(f"Anti-hack (blacklist): {len(operator_results)} clean operators remaining")
    else:
        print("Anti-hack filtering disabled (--no-antihack)")

    if not operator_results:
        print("No operators remaining after antihack filtering.", file=sys.stderr)
        sys.exit(1)

    # Detect or use specified dataset
    if dataset_override and dataset_override != "auto":
        dataset = dataset_override
    else:
        dataset = detect_dataset(operator_results)
    print(f"Dataset: {dataset}")

    if dataset not in DATASET_TOTALS:
        print(f"Warning: Unknown dataset '{dataset}', using KernelGenBench defaults", file=sys.stderr)
        dataset = "KernelGenBench"

    timing = load_timing_data(result_dir)

    # Load token data
    token_info = None
    try:
        from analyze_tokens import scan_workspace_tokens
        token_info = scan_workspace_tokens(result_dir)
    except ImportError:
        # Try relative import
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from analyze_tokens import scan_workspace_tokens
            token_info = scan_workspace_tokens(result_dir)
        except Exception:
            pass

    print_summary(operator_results, result_dir, dataset, timing, token_info)
    save_results(operator_results, result_dir, dataset, timing)


if __name__ == "__main__":
    main()
