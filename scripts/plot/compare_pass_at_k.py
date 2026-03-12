#!/usr/bin/env python3
"""Compare Pass@K results across multiple models."""

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def load_pass_at_k_results(result_path: Path):
    """Load pass_at_k_results.json and extract round progression."""
    with open(result_path) as f:
        data = json.load(f)

    rounds = []
    cumulative = []
    for r in data['rounds_summary']:
        rounds.append(r['round'])
        cumulative.append(r.get('total_passed', r.get('newly_passed', 0)))

    total = data['total_operators']
    return rounds, cumulative, total

def plot_comparison(models_data, output_path: Path):
    """Plot clean comparison of Pass@K progression."""
    plt.figure(figsize=(10, 6))

    colors = ['#2E86AB', '#E63946', '#06A77D', '#F77F00', '#457B9D', '#FF6B9D']

    for idx, (model_name, rounds, cumulative, total) in enumerate(models_data):
        pass_rates = [c / total * 100 for c in cumulative]
        plt.plot(rounds, pass_rates, marker='o', linewidth=2.5, markersize=8,
                label=model_name, color=colors[idx % len(colors)])

    plt.xlabel('Round', fontsize=12, fontweight='bold')
    plt.ylabel('Pass Rate (%)', fontsize=12, fontweight='bold')
    plt.title('Pass@5 Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', required=True, help='Model names')
    parser.add_argument('--paths', nargs='+', required=True, help='Paths to pass_at_k_results.json')
    parser.add_argument('-o', '--output', required=True, help='Output PNG path')
    args = parser.parse_args()

    models_data = []
    for name, path in zip(args.models, args.paths):
        rounds, cumulative, total = load_pass_at_k_results(Path(path))
        models_data.append((name, rounds, cumulative, total))

    plot_comparison(models_data, Path(args.output))

if __name__ == '__main__':
    main()
