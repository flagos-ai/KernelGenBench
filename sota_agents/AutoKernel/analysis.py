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
AutoKernel -- Analysis & visualization of experiment results.

Reads results.tsv, produces:
  - progress.png   : scatter plot of throughput over experiments
  - report.md      : markdown session report
  - terminal output : summary statistics

Usage:
    uv run analysis.py
"""

import json
import os
from datetime import datetime

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results.tsv")
WORKSPACE_RESULTS_DIR = os.path.join(SCRIPT_DIR, "workspace", "results")
PROGRESS_PNG = os.path.join(SCRIPT_DIR, "progress.png")
REPORT_MD = os.path.join(SCRIPT_DIR, "report.md")
BASELINES_PATH = os.path.join(os.path.expanduser("~"), ".cache", "autokernel", "baselines.json")

# Expected TSV columns
EXPECTED_COLUMNS = [
    "experiment", "tag", "kernel_type", "throughput_tflops", "latency_us",
    "pct_peak", "speedup_vs_pytorch", "correctness", "peak_vram_mb", "description",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_single_tsv(path: str) -> pd.DataFrame | None:
    """Load a single TSV file into a DataFrame. Returns None if missing/empty."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path, sep='\t')
    if len(df) == 0:
        return None

    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Convert numeric columns
    for col in ['experiment', 'throughput_tflops', 'latency_us', 'pct_peak',
                'speedup_vs_pytorch', 'peak_vram_mb']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_results(path: str = "results.tsv") -> pd.DataFrame | None:
    """
    Parse results.tsv into a pandas DataFrame, also merging any TSV files
    found in workspace/results/ (written by orchestrate.py).
    Returns None if no data is found.
    """
    frames: list[pd.DataFrame] = []

    # Load the root results.tsv
    root_df = _load_single_tsv(path)
    if root_df is not None:
        frames.append(root_df)

    # Also load all TSV files in workspace/results/ (orchestrate.py output)
    if os.path.isdir(WORKSPACE_RESULTS_DIR):
        for fname in sorted(os.listdir(WORKSPACE_RESULTS_DIR)):
            if fname.endswith(".tsv"):
                ws_path = os.path.join(WORKSPACE_RESULTS_DIR, fname)
                ws_df = _load_single_tsv(ws_path)
                if ws_df is not None:
                    frames.append(ws_df)

    if not frames:
        return None

    df = pd.concat(frames, ignore_index=True)
    if len(df) == 0:
        return None

    # Validate columns against expected set
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing or extra:
        print(f"WARNING: TSV columns do not match expected schema.")
        if missing:
            print(f"  Missing columns: {missing}")
        if extra:
            print(f"  Unexpected columns: {extra}")

    return df


def load_baselines() -> dict | None:
    """Load cached baselines.json if it exists."""
    if not os.path.exists(BASELINES_PATH):
        return None
    with open(BASELINES_PATH, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def classify_row(row) -> str:
    """
    Classify an experiment row (dict or pandas Series) into one of:
      'kept'   -- correctness PASS and tagged as kept / speedup > 1
      'failed' -- correctness FAIL or crash
      'reverted'-- correct but slower (reverted / not kept)
    """
    raw_correctness = row.get("correctness", "")
    correctness = str(raw_correctness).upper() if pd.notna(raw_correctness) else ""
    if correctness in ("FAIL", "CRASH", "ERROR"):
        return "failed"

    # If speedup_vs_pytorch is available and > 1, or if there's no explicit
    # revert indicator, use speedup to decide
    speedup = row.get("speedup_vs_pytorch", "")
    raw_tag = row.get("tag", "")
    tag = str(raw_tag).lower() if pd.notna(raw_tag) else ""

    if tag in ("revert", "reverted", "discard"):
        return "reverted"

    if isinstance(speedup, (int, float)) and pd.notna(speedup):
        if float(speedup) >= 1.0:
            return "kept"
        return "reverted"

    # Default: if correctness is PASS and we can't tell, treat as kept
    if correctness == "PASS":
        return "kept"

    return "reverted"


# ---------------------------------------------------------------------------
# 1. progress.png
# ---------------------------------------------------------------------------

def make_progress_plot(df: pd.DataFrame, baselines: dict | None) -> None:
    """Generate the scatter plot and save to progress.png."""

    fig, ax = plt.subplots(figsize=(12, 6))

    # We plot all kernel types on one chart; if there's only one type that's fine
    xs_kept, ys_kept = [], []
    xs_failed, ys_failed = [], []
    xs_reverted, ys_reverted = [], []

    experiment_nums = []
    throughputs = []

    for i, row in df.iterrows():
        exp_num = row.get("experiment", i + 1)
        if pd.isna(exp_num):
            exp_num = i + 1
        exp_num = float(exp_num)

        tp = row.get("throughput_tflops", 0)
        if pd.isna(tp):
            tp = 0.0
        tp = float(tp)

        experiment_nums.append(exp_num)
        throughputs.append(tp)

        cat = classify_row(row)
        if cat == "kept":
            xs_kept.append(exp_num)
            ys_kept.append(tp)
        elif cat == "failed":
            xs_failed.append(exp_num)
            ys_failed.append(tp)
        else:
            xs_reverted.append(exp_num)
            ys_reverted.append(tp)

    # Scatter dots
    if xs_reverted:
        ax.scatter(xs_reverted, ys_reverted, c="#999999", s=40, alpha=0.6,
                   label="Reverted (correct, slower)", zorder=3, edgecolors="none")
    if xs_failed:
        ax.scatter(xs_failed, ys_failed, c="#e74c3c", s=40, alpha=0.7,
                   label="Failed (FAIL/crash)", zorder=3, edgecolors="none")
    if xs_kept:
        ax.scatter(xs_kept, ys_kept, c="#2ecc71", s=50, alpha=0.85,
                   label="Kept (improved)", zorder=4, edgecolors="none")

    # Running maximum line (research frontier) -- based on kept experiments
    if experiment_nums:
        sorted_pairs = sorted(zip(experiment_nums, throughputs))
        frontier_x, frontier_y = [], []
        running_max = float("-inf")
        for x, y in sorted_pairs:
            if y > running_max:
                running_max = y
            frontier_x.append(x)
            frontier_y.append(running_max)
        ax.plot(frontier_x, frontier_y, color="#27ae60", linewidth=2, alpha=0.8,
                label="Research frontier", zorder=2)

    # PyTorch baseline dashed line
    baseline_tp = _get_baseline_throughput(df, baselines)
    if baseline_tp is not None and baseline_tp > 0:
        ax.axhline(y=baseline_tp, color="#3498db", linestyle="--", linewidth=1.5,
                   alpha=0.7, label=f"PyTorch baseline ({baseline_tp:.1f} TFLOPS)", zorder=1)

    # Annotate top-3 improvements
    if xs_kept and ys_kept:
        top_indices = sorted(range(len(ys_kept)), key=lambda i: ys_kept[i], reverse=True)[:3]
        for rank, idx in enumerate(top_indices):
            ax.annotate(
                f"#{rank + 1}: {ys_kept[idx]:.2f}",
                xy=(xs_kept[idx], ys_kept[idx]),
                xytext=(10, 10 + rank * 15),
                textcoords="offset points",
                fontsize=8,
                color="#27ae60",
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#27ae60", lw=0.8),
                zorder=5,
            )

    # Styling
    ax.set_xlabel("Experiment #", fontsize=11)
    ax.set_ylabel("Throughput (TFLOPS)", fontsize=11)
    ax.set_title("AutoKernel -- Optimization Progress", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # Ensure y-axis starts at 0 if data allows
    ymin = min(throughputs) if throughputs else 0
    if ymin >= 0:
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(PROGRESS_PNG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PROGRESS_PNG}")


def _get_baseline_throughput(df: pd.DataFrame, baselines: dict | None) -> float | None:
    """
    Determine baseline throughput from the first row in results or from
    baselines.json.
    """
    # Try first row
    if df is not None and len(df) > 0 and "throughput_tflops" in df.columns:
        first_tp = df.iloc[0]["throughput_tflops"]
        if pd.notna(first_tp) and float(first_tp) > 0:
            return float(first_tp)

    # Fallback: baselines.json -- pick the best throughput across configs
    if baselines:
        best = 0.0
        for entry in baselines.values():
            tp = entry.get("throughput_tflops", 0)
            if tp > best:
                best = tp
        if best > 0:
            return best

    return None


# ---------------------------------------------------------------------------
# 2. Terminal summary
# ---------------------------------------------------------------------------

def print_terminal_summary(df: pd.DataFrame, baselines: dict | None) -> None:
    """Print a concise summary of the experiment session to stdout."""

    print()
    print("=" * 60)
    print("  AutoKernel -- Session Summary")
    print("=" * 60)

    # Group by kernel_type
    if "kernel_type" in df.columns:
        kernel_types = sorted(df["kernel_type"].fillna("unknown").unique())
    else:
        kernel_types = ["unknown"]

    for kt in kernel_types:
        kt_df = df[df["kernel_type"].fillna("unknown") == kt] if "kernel_type" in df.columns else df
        print(f"\n  Kernel type: {kt}")
        print(f"  {'=' * 40}")

        # Classify
        n_total = len(kt_df)
        classifications = kt_df.apply(classify_row, axis=1)
        n_kept = (classifications == "kept").sum()
        n_failed = (classifications == "failed").sum()
        n_reverted = (classifications == "reverted").sum()

        keep_rate = (n_kept / n_total * 100) if n_total > 0 else 0
        crash_rate = (n_failed / n_total * 100) if n_total > 0 else 0

        # Throughput stats
        valid_tps = kt_df["throughput_tflops"].dropna()
        valid_tps = valid_tps[valid_tps > 0]

        baseline_tp = _get_baseline_throughput(kt_df, baselines)
        best_tp = float(valid_tps.max()) if len(valid_tps) > 0 else None

        if baseline_tp:
            print(f"  Baseline throughput:    {baseline_tp:.2f} TFLOPS")
        if best_tp:
            print(f"  Current best:          {best_tp:.2f} TFLOPS")
        if baseline_tp and best_tp and baseline_tp > 0:
            speedup = best_tp / baseline_tp
            print(f"  Total speedup:         {speedup:.2f}x vs PyTorch")

        print(f"  Experiments:           {n_total}")
        print(f"  Kept:                  {n_kept} ({keep_rate:.0f}%)")
        print(f"  Reverted:              {n_reverted}")
        print(f"  Failed/crashed:        {n_failed} ({crash_rate:.0f}%)")

        # Top 5 improvements by throughput delta over baseline
        if baseline_tp and baseline_tp > 0:
            deltas = []
            for idx, row in kt_df.iterrows():
                tp = row.get("throughput_tflops", 0)
                if pd.notna(tp) and float(tp) > 0 and classify_row(row) == "kept":
                    delta = float(tp) - baseline_tp
                    deltas.append((delta, row))
            deltas.sort(key=lambda x: x[0], reverse=True)

            if deltas:
                print(f"\n  Top 5 improvements:")
                for rank, (delta, r) in enumerate(deltas[:5], 1):
                    desc = r.get("description", "no description")
                    tp = float(r.get("throughput_tflops", 0))
                    sign = "+" if delta >= 0 else ""
                    print(f"    {rank}. {tp:.2f} TFLOPS ({sign}{delta:.2f}) -- {desc}")

        # Roofline position
        kept_mask = classifications == "kept"
        if "pct_peak" in kt_df.columns:
            kept_pct = kt_df.loc[kept_mask, "pct_peak"].dropna()
            kept_pct = kept_pct[kept_pct > 0]
            if len(kept_pct) > 0:
                best_pct = float(kept_pct.max())
                print(f"\n  Roofline position:     {best_pct:.1f}% of peak")

    print()
    print("=" * 60)
    print()


# ---------------------------------------------------------------------------
# 3. report.md
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame, baselines: dict | None) -> None:
    """Generate a markdown report summarizing the session."""

    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("# AutoKernel Session Report")
    lines.append("")
    lines.append(f"Generated: {timestamp}")
    lines.append("")

    # Group by kernel type
    if "kernel_type" in df.columns:
        kernel_types = sorted(df["kernel_type"].fillna("unknown").unique())
    else:
        kernel_types = ["unknown"]

    for kt in kernel_types:
        kt_df = df[df["kernel_type"].fillna("unknown") == kt] if "kernel_type" in df.columns else df
        classifications = kt_df.apply(classify_row, axis=1)

        lines.append(f"## Kernel: {kt}")
        lines.append("")

        # Summary stats
        n_total = len(kt_df)
        n_kept = int((classifications == "kept").sum())
        n_failed = int((classifications == "failed").sum())
        n_reverted = int((classifications == "reverted").sum())

        baseline_tp = _get_baseline_throughput(kt_df, baselines)
        valid_tps = kt_df["throughput_tflops"].dropna()
        valid_tps = valid_tps[valid_tps > 0]
        best_tp = float(valid_tps.max()) if len(valid_tps) > 0 else None

        lines.append("### Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Total experiments | {n_total} |")
        lines.append(f"| Kept | {n_kept} |")
        lines.append(f"| Reverted | {n_reverted} |")
        lines.append(f"| Failed | {n_failed} |")
        if baseline_tp:
            lines.append(f"| Baseline throughput | {baseline_tp:.2f} TFLOPS |")
        if best_tp:
            lines.append(f"| Best throughput | {best_tp:.2f} TFLOPS |")
        if baseline_tp and best_tp and baseline_tp > 0:
            lines.append(f"| Speedup vs PyTorch | {best_tp / baseline_tp:.2f}x |")
        lines.append("")

        # Key discoveries (kept experiments)
        kept_df = kt_df[classifications == "kept"]
        if len(kept_df) > 0:
            lines.append("### Key Discoveries (Kept)")
            lines.append("")
            for _, r in kept_df.iterrows():
                exp = r.get("experiment", "?")
                tp = r.get("throughput_tflops", 0)
                desc = r.get("description", "no description")
                speedup = r.get("speedup_vs_pytorch", "N/A")
                if pd.notna(speedup) and isinstance(speedup, (int, float)):
                    speedup = f"{float(speedup):.2f}x"
                tp_val = float(tp) if pd.notna(tp) else 0
                lines.append(f"- **Exp {exp}**: {tp_val:.2f} TFLOPS (speedup: {speedup}) -- {desc}")
            lines.append("")

        # Failed experiments
        failed_df = kt_df[classifications == "failed"]
        if len(failed_df) > 0:
            lines.append("### Failed Experiments")
            lines.append("")
            for _, r in failed_df.iterrows():
                exp = r.get("experiment", "?")
                desc = r.get("description", "no description")
                correctness = r.get("correctness", "unknown")
                lines.append(f"- **Exp {exp}** [{correctness}]: {desc}")
            lines.append("")

        # Reverted experiments
        reverted_df = kt_df[classifications == "reverted"]
        if len(reverted_df) > 0:
            lines.append("### Reverted Experiments (Correct but Slower)")
            lines.append("")
            for _, r in reverted_df.iterrows():
                exp = r.get("experiment", "?")
                tp = r.get("throughput_tflops", 0)
                desc = r.get("description", "no description")
                if pd.notna(tp):
                    lines.append(f"- **Exp {exp}**: {float(tp):.2f} TFLOPS -- {desc}")
                else:
                    lines.append(f"- **Exp {exp}**: {desc}")
            lines.append("")

        # Current state of optimization
        lines.append("### Current Optimization State")
        lines.append("")
        if best_tp and baseline_tp and baseline_tp > 0:
            pct_gain = (best_tp - baseline_tp) / baseline_tp * 100
            lines.append(f"The best kernel achieves **{best_tp:.2f} TFLOPS**, which is "
                         f"**{pct_gain:+.1f}%** relative to the PyTorch baseline ({baseline_tp:.2f} TFLOPS).")
        elif best_tp:
            lines.append(f"The best kernel achieves **{best_tp:.2f} TFLOPS**.")
        else:
            lines.append("No successful runs have been completed yet.")
        lines.append("")

        # Roofline
        if "pct_peak" in kt_df.columns:
            kept_pct = kt_df.loc[classifications == "kept", "pct_peak"].dropna()
            kept_pct = kept_pct[kept_pct > 0]
            if len(kept_pct) > 0:
                best_pct = float(kept_pct.max())
                lines.append(f"Current roofline utilization: **{best_pct:.1f}%** of theoretical peak.")
                lines.append("")

        # Suggestions
        lines.append("### Suggestions for Next Session")
        lines.append("")

        suggestions = _generate_suggestions(kt_df, baseline_tp, best_tp, n_failed, n_total)
        for s in suggestions:
            lines.append(f"- {s}")
        lines.append("")

    # Write file
    report_text = "\n".join(lines)
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Saved: {REPORT_MD}")


def _generate_suggestions(
    df: pd.DataFrame,
    baseline_tp: float | None,
    best_tp: float | None,
    n_failed: int,
    n_total: int,
) -> list[str]:
    """Generate actionable suggestions based on experiment history."""

    suggestions = []

    if n_total == 0:
        return ["Run some experiments first to generate suggestions."]

    # High crash rate
    if n_total > 0 and n_failed / n_total > 0.4:
        suggestions.append(
            "High crash/failure rate ({:.0f}%). Consider more conservative changes or "
            "better input validation in the kernel.".format(n_failed / n_total * 100)
        )

    # Speedup analysis
    if baseline_tp and best_tp and baseline_tp > 0:
        speedup = best_tp / baseline_tp
        if speedup < 1.1:
            suggestions.append(
                "Speedup over PyTorch is modest (<1.1x). Consider trying: "
                "autotuning over block sizes, persistent kernels, or split-K strategies."
            )
        elif speedup < 1.5:
            suggestions.append(
                "Decent speedup achieved. Next steps: try software pipelining, "
                "warp specialization, or TMA-based data movement."
            )
        else:
            suggestions.append(
                "Strong speedup achieved. Consider: fine-grained autotuning across "
                "more size configurations, or targeting remaining bottlenecks with profiling."
            )

    # Plateau detection: if last N experiments were all reverted
    last_5 = df.tail(5)
    if len(last_5) >= 5:
        last_5_cats = last_5.apply(classify_row, axis=1)
        if all(c in ("reverted", "failed") for c in last_5_cats):
            suggestions.append(
                "Last 5 experiments were all reverted or failed -- possible plateau. "
                "Try a fundamentally different approach (different algorithm, memory layout, "
                "or kernel fusion strategy)."
            )

    # Memory observations
    if "peak_vram_mb" in df.columns:
        classifications = df.apply(classify_row, axis=1)
        kept_vrams = df.loc[classifications == "kept", "peak_vram_mb"].dropna()
        kept_vrams = kept_vrams[kept_vrams > 0]
        if len(kept_vrams) > 0 and float(kept_vrams.max()) > 10000:
            suggestions.append(
                f"Peak VRAM usage is high ({float(kept_vrams.max()):.0f} MB). Consider memory-efficient "
                f"techniques if you need headroom for larger problem sizes."
            )

    if not suggestions:
        suggestions.append(
            "Continue iterating. Try systematic autotuning of block sizes and "
            "explore Triton-specific optimizations (e.g., num_warps, num_stages)."
        )

    return suggestions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Load results
    df = load_results(RESULTS_PATH)

    if df is None:
        print("No results.tsv found. Run some experiments first.")
        return

    if len(df) == 0:
        print("No experiments yet (results.tsv contains only the header).")
        return

    baselines = load_baselines()

    # Single-row edge case
    if len(df) == 1:
        print(f"Only 1 experiment recorded (baseline).")
        row = df.iloc[0]
        tp = row.get("throughput_tflops", "N/A")
        desc = row.get("description", "no description")
        if pd.notna(tp) and isinstance(tp, (int, float)):
            print(f"  Throughput: {tp:.2f} TFLOPS -- {desc}")
        else:
            print(f"  Throughput: {tp} -- {desc}")
        print("\nRun more experiments to generate full analysis.")
        # Still generate what we can
        make_progress_plot(df, baselines)
        generate_report(df, baselines)
        return

    # Full analysis
    make_progress_plot(df, baselines)
    print_terminal_summary(df, baselines)
    generate_report(df, baselines)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
