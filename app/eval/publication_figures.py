#!/usr/bin/env python3
"""
Generate publication-quality figures for MixSense evaluation.

Figures:
1. Predicted vs True concentration scatter plot
2. Error vs difficulty level bar chart
3. Example spectrum reconstruction comparison
4. Domain gap illustration

Usage:
    python -m app.eval.publication_figures --results results.json --output figures/
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Try to import matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, figure generation disabled")


# ============================================================================
# Figure Style Configuration
# ============================================================================

FIGURE_STYLE = {
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
}

COLORS = {
    "easy": "#2ecc71",      # Green
    "medium": "#f1c40f",    # Yellow
    "hard": "#e74c3c",      # Red
    "primary": "#3498db",   # Blue
    "secondary": "#9b59b6", # Purple
    "mixture": "#2c3e50",   # Dark blue
    "reference": "#95a5a6", # Gray
}


def setup_style():
    """Apply publication style to matplotlib."""
    if HAS_MATPLOTLIB:
        plt.rcParams.update(FIGURE_STYLE)


# ============================================================================
# Figure 1: Predicted vs True Scatter Plot
# ============================================================================

def plot_predicted_vs_true(
    results: List[Dict],
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """
    Scatter plot of predicted vs true concentrations.

    Diagonal line = perfect prediction
    Color = difficulty level
    """
    if not HAS_MATPLOTLIB:
        return None

    setup_style()

    fig, ax = plt.subplots(figsize=(5, 5))

    # Collect all points
    for result in results:
        difficulty = result.get("difficulty", "medium")
        color = COLORS.get(difficulty, COLORS["primary"])

        for comp, true_val in result["ground_truth"].items():
            pred_val = result["predicted"].get(comp, 0.0)
            ax.scatter(true_val, pred_val, c=color, alpha=0.7, s=50, edgecolors='white', linewidth=0.5)

    # Perfect prediction line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect')

    # Labels
    ax.set_xlabel("True Concentration")
    ax.set_ylabel("Predicted Concentration")
    ax.set_title("Deconvolution Accuracy")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["easy"],
               markersize=8, label='Easy'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["medium"],
               markersize=8, label='Medium'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS["hard"],
               markersize=8, label='Hard'),
        Line2D([0], [0], linestyle='--', color='k', label='Perfect'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


# ============================================================================
# Figure 2: Error by Difficulty Bar Chart
# ============================================================================

def plot_error_by_difficulty(
    summary_by_difficulty: Dict[str, Dict],
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """
    Bar chart showing MAE at different difficulty levels.
    """
    if not HAS_MATPLOTLIB:
        return None

    setup_style()

    fig, ax = plt.subplots(figsize=(5, 4))

    difficulties = ["easy", "medium", "hard"]
    x = np.arange(len(difficulties))
    width = 0.6

    maes = []
    stds = []
    colors = []

    for d in difficulties:
        if d in summary_by_difficulty:
            maes.append(summary_by_difficulty[d]["mae_mean"])
            stds.append(summary_by_difficulty[d].get("mae_std", 0))
        else:
            maes.append(0)
            stds.append(0)
        colors.append(COLORS[d])

    bars = ax.bar(x, maes, width, yerr=stds, capsize=5, color=colors, edgecolor='white', linewidth=1)

    ax.set_ylabel("Mean Absolute Error (MAE)")
    ax.set_xlabel("Difficulty Level")
    ax.set_title("Deconvolution Error vs Domain Gap")
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in difficulties])
    ax.set_ylim(0, max(maes) * 1.3 if maes else 0.1)

    # Add value labels on bars
    for bar, mae in zip(bars, maes):
        height = bar.get_height()
        ax.annotate(f'{mae:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


# ============================================================================
# Figure 3: Example Spectrum with Reconstruction
# ============================================================================

def plot_spectrum_reconstruction(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    reconstructed_intensity: List[float],
    ground_truth: Dict[str, float],
    predicted: Dict[str, float],
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """
    Plot mixture spectrum vs reconstructed spectrum.
    """
    if not HAS_MATPLOTLIB:
        return None

    setup_style()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1])

    ppm = np.array(mixture_ppm)
    mix = np.array(mixture_intensity)
    recon = np.array(reconstructed_intensity)

    # Main plot: spectra
    ax1.plot(ppm, mix, color=COLORS["mixture"], linewidth=1, label="Mixture (exp.)")
    ax1.plot(ppm, recon, color=COLORS["primary"], linewidth=1, linestyle='--', label="Reconstructed")
    ax1.fill_between(ppm, mix, alpha=0.2, color=COLORS["mixture"])

    ax1.set_xlim(ppm.max(), ppm.min())  # NMR convention: high to low
    ax1.set_xlabel("Chemical Shift (ppm)")
    ax1.set_ylabel("Intensity")
    ax1.legend(loc='upper left')

    # Residual plot
    residual = mix - recon
    ax2.plot(ppm, residual, color='gray', linewidth=0.5)
    ax2.fill_between(ppm, residual, alpha=0.3, color='gray')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlim(ppm.max(), ppm.min())
    ax2.set_xlabel("Chemical Shift (ppm)")
    ax2.set_ylabel("Residual")

    # Add composition annotation
    gt_str = ", ".join([f"{k}: {v:.0%}" for k, v in ground_truth.items()])
    pred_str = ", ".join([f"{k}: {v:.0%}" for k, v in predicted.items()])
    ax1.set_title(f"Ground truth: {gt_str}\nPredicted: {pred_str}", fontsize=10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


# ============================================================================
# Figure 4: Domain Gap Illustration
# ============================================================================

def plot_domain_gap_illustration(
    original_peaks: List[float],
    perturbed_peaks: List[float],
    difficulty: str = "medium",
    output_path: Optional[str] = None,
    show: bool = False,
) -> Optional[plt.Figure]:
    """
    Illustrate the domain gap concept with original vs perturbed peaks.
    """
    if not HAS_MATPLOTLIB:
        return None

    setup_style()

    fig, ax = plt.subplots(figsize=(6, 3))

    # Generate spectra
    ppm_grid = np.linspace(0, 10, 1000)

    def spectrum_from_peaks(peaks, width=0.02):
        spec = np.zeros_like(ppm_grid)
        for p in peaks:
            spec += (width**2) / ((ppm_grid - p)**2 + width**2)
        return spec / max(spec) if max(spec) > 0 else spec

    orig_spec = spectrum_from_peaks(original_peaks)
    pert_spec = spectrum_from_peaks(perturbed_peaks)

    ax.plot(ppm_grid, orig_spec, color=COLORS["reference"], linewidth=1.5, label="Database reference")
    ax.plot(ppm_grid, pert_spec, color=COLORS["mixture"], linewidth=1.5, label="Experimental (simulated)")

    # Highlight shifts with arrows
    for orig, pert in zip(original_peaks, perturbed_peaks):
        if abs(orig - pert) > 0.01:
            ax.annotate('', xy=(pert, 0.5), xytext=(orig, 0.5),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1))

    ax.set_xlim(ppm_grid.max(), ppm_grid.min())
    ax.set_xlabel("Chemical Shift (ppm)")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Domain Gap Illustration ({difficulty.capitalize()} difficulty)")
    ax.legend(loc='upper left')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"Saved: {output_path}")

    if show:
        plt.show()

    return fig


# ============================================================================
# Generate All Figures from Results
# ============================================================================

def generate_all_figures(
    results_path: str,
    output_dir: str,
    show: bool = False,
):
    """
    Generate all publication figures from evaluation results.
    """
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, cannot generate figures")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Load results
    with open(results_path, 'r') as f:
        data = json.load(f)

    print(f"Generating figures from: {results_path}")
    print(f"Output directory: {output_dir}")

    # Figure 1: Scatter plot
    if "results" in data:
        plot_predicted_vs_true(
            data["results"],
            output_path=os.path.join(output_dir, "fig1_scatter.png"),
            show=show
        )

    # Figure 2: Error by difficulty
    if "by_category" in data or isinstance(data, dict):
        # Check if this is a difficulty comparison result
        if all(k in data for k in ["easy", "medium", "hard"]):
            plot_error_by_difficulty(
                data,
                output_path=os.path.join(output_dir, "fig2_difficulty.png"),
                show=show
            )

    print("\nFigures generated successfully!")


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--results", "-r", type=str, required=True,
                        help="Path to evaluation results JSON")
    parser.add_argument("--output", "-o", type=str, default="figures",
                        help="Output directory for figures")
    parser.add_argument("--show", action="store_true",
                        help="Show figures interactively")

    args = parser.parse_args()

    generate_all_figures(
        results_path=args.results,
        output_dir=args.output,
        show=args.show
    )


if __name__ == "__main__":
    main()
