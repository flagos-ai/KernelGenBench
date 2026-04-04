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
    python scripts/analyze/analyze_speedup_v6.py <result_dir> [--dataset auto|v2_1|KernelGenBench] [--no-antihack]
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


def load_from_verification_dir(result_dir: Path) -> Dict[str, Dict]:
    """Load results from verification directory.

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

    for log_dir in log_dirs:
        round_num = int(log_dir.name.split("_")[1])
        result_file = log_dir / "result.json"

        if not result_file.exists():
            continue

        try:
            with open(result_file, 'r') as f:
                results = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {result_file}: {e}", file=sys.stderr)
            continue

        for item in results:
            op_name = item.get("op_name")
            if not op_name:
                continue

            if not item.get("success"):
                continue

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
        "median": median(values),
        "iqm": iqm(values),
        "trimmed_mean_10": trimmed_mean(values, 0.1),
        "min": min(values),
        "max": max(values),
    }


# ── Console output ───────────────────────────────────────────────────────────

def print_summary(operator_results: Dict[str, Dict], result_dir: Path, dataset: str):
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
    print(f"{'Category':<15} {'Passed':>8} {'Total':>8} {'GeoMean':>10} {'Median':>10} {'IQM':>10} {'ArithMean':>10} {'Min':>10} {'Max':>10}")
    print("-" * 110)

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
            print(f"{label:<15} {0:>8} {total:>8} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
            continue

        print(f"{label:<15} {s['count']:>8} {total:>8} {s['geo_mean']:>10.4f} {s['median']:>10.4f} {s['iqm']:>10.4f} {s['arith_mean']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f}")

    print("-" * 110)
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


# ── Markdown output ──────────────────────────────────────────────────────────

def generate_markdown(operator_results: Dict[str, Dict], result_dir: Path, dataset: str) -> str:
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
    lines.append("| Category | Passed | Total | GeoMean | Median | IQM | ArithMean | Min | Max |")
    lines.append("|----------|--------|-------|---------|--------|-----|-----------|-----|-----|")

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
            lines.append(f"| {label} | 0 | {total} | - | - | - | - | - | - |")
            continue
        lines.append(f"| {label} | {s['count']} | {total} | {s['geo_mean']:.4f} | {s['median']:.4f} | {s['iqm']:.4f} | {s['arith_mean']:.4f} | {s['min']:.4f} | {s['max']:.4f} |")

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

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def save_results(operator_results: Dict[str, Dict], result_dir: Path, dataset: str):
    """Save results to markdown file."""
    md_content = generate_markdown(operator_results, result_dir, dataset)
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

    print_summary(operator_results, result_dir, dataset)
    save_results(operator_results, result_dir, dataset)


if __name__ == "__main__":
    main()
