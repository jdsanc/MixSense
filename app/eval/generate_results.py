#!/usr/bin/env python3
"""
Generate publication-quality tables and figures from benchmark results.

Usage:
    python -m app.eval.generate_results
    python -m app.eval.generate_results --results results/baseline_comparison_*.json
    python -m app.eval.generate_results --output figures/
"""

import os
import sys
import json
import argparse
import glob
from typing import Dict, List, Optional
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, will generate text tables only")


# ============================================================================
# Table Generation (Text/Markdown/LaTeX)
# ============================================================================

def generate_comparison_table(
    results: Dict,
    format: str = "markdown",
) -> str:
    """
    Generate a comparison table from benchmark results.

    Args:
        results: Benchmark results dict with 'by_difficulty' key
        format: 'markdown', 'latex', or 'text'

    Returns:
        Formatted table string
    """
    if "by_difficulty" not in results:
        return "Error: Results don't contain 'by_difficulty' key"

    by_diff = results["by_difficulty"]
    difficulties = ["easy", "medium", "hard"]

    # Get all methods from first difficulty level
    first_diff = list(by_diff.values())[0]
    methods = list(first_diff.keys())

    # Build table
    if format == "markdown":
        lines = []
        # Header
        header = "| Method | " + " | ".join(d.capitalize() for d in difficulties) + " |"
        separator = "|" + "|".join(["---"] * (len(difficulties) + 1)) + "|"
        lines.append(header)
        lines.append(separator)

        # Data rows
        for method in methods:
            row = f"| {method} |"
            for diff in difficulties:
                if diff in by_diff and method in by_diff[diff]:
                    mae = by_diff[diff][method]["mae_mean"]
                    row += f" {mae:.4f} |"
                else:
                    row += " N/A |"
            lines.append(row)

        return "\n".join(lines)

    elif format == "latex":
        lines = []
        lines.append("\\begin{table}[h]")
        lines.append("\\centering")
        lines.append("\\caption{Deconvolution MAE by Method and Difficulty}")
        lines.append("\\begin{tabular}{l" + "c" * len(difficulties) + "}")
        lines.append("\\toprule")
        lines.append("Method & " + " & ".join(d.capitalize() for d in difficulties) + " \\\\")
        lines.append("\\midrule")

        for method in methods:
            row = f"{method} &"
            values = []
            for diff in difficulties:
                if diff in by_diff and method in by_diff[diff]:
                    mae = by_diff[diff][method]["mae_mean"]
                    values.append(f"{mae:.4f}")
                else:
                    values.append("N/A")
            row += " & ".join(values) + " \\\\"
            lines.append(row)

        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\label{tab:baseline_comparison}")
        lines.append("\\end{table}")

        return "\n".join(lines)

    else:  # text
        lines = []
        # Header
        col_width = 15
        header = f"{'Method':<{col_width}}" + "".join(f"{d.capitalize():<12}" for d in difficulties)
        lines.append(header)
        lines.append("-" * len(header))

        for method in methods:
            row = f"{method:<{col_width}}"
            for diff in difficulties:
                if diff in by_diff and method in by_diff[diff]:
                    mae = by_diff[diff][method]["mae_mean"]
                    row += f"{mae:<12.4f}"
                else:
                    row += f"{'N/A':<12}"
            lines.append(row)

        return "\n".join(lines)


def generate_ranking_table(results: Dict, difficulty: str = "medium") -> str:
    """Generate a ranked table for a specific difficulty level."""
    if "by_difficulty" not in results:
        return "Error: Results don't contain 'by_difficulty' key"

    if difficulty not in results["by_difficulty"]:
        return f"Error: Difficulty '{difficulty}' not found"

    data = results["by_difficulty"][difficulty]

    # Sort by MAE
    ranked = sorted(data.items(), key=lambda x: x[1]["mae_mean"])

    lines = []
    lines.append(f"\n## Ranking ({difficulty.capitalize()} Difficulty)")
    lines.append("")
    lines.append("| Rank | Method | MAE | RMSE |")
    lines.append("|------|--------|-----|------|")

    for i, (method, stats) in enumerate(ranked, 1):
        mae = stats["mae_mean"]
        rmse = stats["rmse_mean"]
        lines.append(f"| {i} | {method} | {mae:.4f} | {rmse:.4f} |")

    return "\n".join(lines)


# ============================================================================
# Figure Generation
# ============================================================================

def plot_method_comparison_bar(
    results: Dict,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """
    Generate bar chart comparing methods across difficulty levels.
    """
    if not HAS_MATPLOTLIB:
        return None

    by_diff = results.get("by_difficulty", {})
    if not by_diff:
        print("No difficulty data found")
        return None

    difficulties = ["easy", "medium", "hard"]
    methods = list(list(by_diff.values())[0].keys())

    # Remove random from display (it's too high and compresses the scale)
    display_methods = [m for m in methods if m != "random"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(display_methods))
    width = 0.25

    colors = {"easy": "#2ecc71", "medium": "#f1c40f", "hard": "#e74c3c"}

    for i, diff in enumerate(difficulties):
        if diff not in by_diff:
            continue

        maes = []
        stds = []
        for method in display_methods:
            if method in by_diff[diff]:
                maes.append(by_diff[diff][method]["mae_mean"])
                stds.append(by_diff[diff][method].get("mae_std", 0))
            else:
                maes.append(0)
                stds.append(0)

        offset = (i - 1) * width
        bars = ax.bar(x + offset, maes, width, yerr=stds,
                     label=diff.capitalize(), color=colors[diff],
                     capsize=3, alpha=0.8)

    ax.set_xlabel("Method", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
    ax.set_title("Deconvolution Accuracy: Method Comparison", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(display_methods, rotation=45, ha='right')
    ax.legend(title="Difficulty")
    ax.set_ylim(0, None)

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    plt.close()
    return output_path


def plot_difficulty_comparison(
    results: Dict,
    method: str = "nnls",
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """
    Plot how a specific method performs across difficulties.
    """
    if not HAS_MATPLOTLIB:
        return None

    by_diff = results.get("by_difficulty", {})
    difficulties = ["easy", "medium", "hard"]

    maes = []
    stds = []

    for diff in difficulties:
        if diff in by_diff and method in by_diff[diff]:
            maes.append(by_diff[diff][method]["mae_mean"])
            stds.append(by_diff[diff][method].get("mae_std", 0))
        else:
            maes.append(0)
            stds.append(0)

    fig, ax = plt.subplots(figsize=(6, 4))

    colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
    x = range(len(difficulties))

    bars = ax.bar(x, maes, yerr=stds, color=colors, capsize=5, alpha=0.8)

    ax.set_xlabel("Difficulty Level", fontsize=12)
    ax.set_ylabel("Mean Absolute Error (MAE)", fontsize=12)
    ax.set_title(f"Performance of {method.upper()} Across Difficulty Levels", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulties])

    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax.annotate(f'{mae:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    plt.close()
    return output_path


def plot_heatmap(
    results: Dict,
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[str]:
    """
    Generate heatmap of MAE by method and difficulty.
    """
    if not HAS_MATPLOTLIB:
        return None

    by_diff = results.get("by_difficulty", {})
    difficulties = ["easy", "medium", "hard"]
    methods = list(list(by_diff.values())[0].keys())

    # Build matrix
    matrix = []
    for method in methods:
        row = []
        for diff in difficulties:
            if diff in by_diff and method in by_diff[diff]:
                row.append(by_diff[diff][method]["mae_mean"])
            else:
                row.append(np.nan)
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')

    # Labels
    ax.set_xticks(range(len(difficulties)))
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("MAE (lower is better)", rotation=-90, va="bottom")

    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(difficulties)):
            val = matrix[i, j]
            if not np.isnan(val):
                text = ax.text(j, i, f'{val:.3f}',
                              ha="center", va="center", color="black", fontsize=9)

    ax.set_title("Deconvolution MAE Heatmap", fontsize=14)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    plt.close()
    return output_path


# ============================================================================
# Main
# ============================================================================

def generate_all(
    results_path: str,
    output_dir: str = "figures",
    show: bool = False,
):
    """Generate all tables and figures from results file."""

    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("GENERATING PUBLICATION RESULTS")
    print("=" * 60)
    print(f"\nInput: {results_path}")
    print(f"Output: {output_dir}/")

    # Generate tables
    print("\n--- TABLES ---\n")

    # Markdown table
    md_table = generate_comparison_table(results, format="markdown")
    print("### Comparison Table (Markdown)\n")
    print(md_table)

    # Save markdown
    md_path = os.path.join(output_dir, "comparison_table.md")
    with open(md_path, 'w') as f:
        f.write("# Baseline Comparison Results\n\n")
        f.write("## MAE by Method and Difficulty\n\n")
        f.write(md_table)
        f.write("\n\n")
        f.write(generate_ranking_table(results, "easy"))
        f.write("\n\n")
        f.write(generate_ranking_table(results, "medium"))
        f.write("\n\n")
        f.write(generate_ranking_table(results, "hard"))
    print(f"\nSaved: {md_path}")

    # LaTeX table
    latex_table = generate_comparison_table(results, format="latex")
    latex_path = os.path.join(output_dir, "comparison_table.tex")
    with open(latex_path, 'w') as f:
        f.write(latex_table)
    print(f"Saved: {latex_path}")

    # Generate figures
    if HAS_MATPLOTLIB:
        print("\n--- FIGURES ---\n")

        plot_method_comparison_bar(
            results,
            output_path=os.path.join(output_dir, "method_comparison.png"),
            show=show
        )

        plot_difficulty_comparison(
            results,
            method="nnls",
            output_path=os.path.join(output_dir, "nnls_by_difficulty.png"),
            show=show
        )

        plot_heatmap(
            results,
            output_path=os.path.join(output_dir, "mae_heatmap.png"),
            show=show
        )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate tables and figures from benchmark results")
    parser.add_argument("--results", "-r", type=str,
                        help="Path to results JSON (defaults to latest baseline_comparison)")
    parser.add_argument("--output", "-o", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--show", action="store_true",
                        help="Show figures interactively")

    args = parser.parse_args()

    # Find results file
    if args.results:
        results_path = args.results
    else:
        # Find latest baseline comparison
        pattern = "results/baseline_comparison_*.json"
        files = sorted(glob.glob(pattern))
        if not files:
            print(f"No results found matching: {pattern}")
            print("Run: python -m app.eval.run_all_benchmarks --realistic-only")
            return
        results_path = files[-1]
        print(f"Using latest results: {results_path}")

    generate_all(
        results_path=results_path,
        output_dir=args.output,
        show=args.show
    )


if __name__ == "__main__":
    main()
