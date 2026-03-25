"""
Plot and compare NMR spectra: single spectra, overlays, and stacked time-series.

Usage:
    # Overlay multiple spectra
    python nmr_plot.py spectrum1.csv spectrum2.csv --labels "Mixture" "Ref A" \\
        --title "Comparison" --output overlay.png

    # Stacked time-series plot
    python nmr_plot.py t0.csv t1.csv t2.csv --stacked \\
        --labels 0min 30min 60min --output time_series.png

Requirements:
    - Environment: mixsense (uv sync)
    - Required packages: numpy, matplotlib
"""

import argparse
import pathlib
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from nmr_io import load_spectrum


def plot_overlay(
    spectra: list,
    labels: list,
    title: str,
    out_path: pathlib.Path,
    ppm_range: tuple = None,
) -> None:
    """
    Overlay multiple NMR spectra on a single axes.

    Args:
        spectra: List of (ppm, intensity) tuples.
        labels: Legend labels for each spectrum.
        title: Plot title.
        out_path: Output file path (.png or .pdf).
        ppm_range: Optional (ppm_min, ppm_max) tuple to restrict the x-axis.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, ((ppm, intensity), label) in enumerate(zip(spectra, labels)):
        ax.plot(ppm, intensity, linewidth=0.9, label=label, color=f"C{i}")
    ax.invert_xaxis()
    ax.set_xlabel("Chemical shift (ppm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title(title)
    ax.legend()
    if ppm_range is not None:
        ax.set_xlim(ppm_range[1], ppm_range[0])  # inverted axis
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stacked(
    spectra: list,
    labels: list,
    title: str,
    out_path: pathlib.Path,
    ppm_range: tuple = None,
) -> None:
    """
    Stack NMR spectra vertically (offset), useful for time-series visualization.

    Args:
        spectra: List of (ppm, intensity) tuples (time-ordered).
        labels: Labels for each spectrum (shown on y-axis).
        title: Plot title.
        out_path: Output file path.
        ppm_range: Optional (ppm_min, ppm_max) to restrict the x-axis.
    """
    n = len(spectra)
    fig, ax = plt.subplots(figsize=(10, 2 * n))
    for i, ((ppm, intensity), label) in enumerate(zip(spectra, labels)):
        offset = i * (intensity.max() - intensity.min()) * 1.2
        ax.plot(ppm, intensity + offset, linewidth=0.8, color=f"C{i % 10}", label=label)
        ax.text(ppm.max() + 0.05, offset + intensity.mean(), label, fontsize=8, va="center")
    ax.invert_xaxis()
    ax.set_xlabel("Chemical shift (ppm)")
    ax.set_title(title)
    ax.set_yticks([])
    if ppm_range is not None:
        ax.set_xlim(ppm_range[1], ppm_range[0])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Plot and compare NMR spectra.")
    ap.add_argument("spectra", nargs="+", help="Input .csv or .xy spectrum files")
    ap.add_argument("--labels", nargs="+", help="Legend labels (defaults to file stems)")
    ap.add_argument("--title", default="NMR Spectra", help="Plot title")
    ap.add_argument("--output", default="nmr_plot.png", help="Output file path")
    ap.add_argument(
        "--stacked", action="store_true",
        help="Use stacked (offset) layout instead of overlay",
    )
    ap.add_argument("--ppm_min", type=float, default=None, help="Minimum ppm to display")
    ap.add_argument("--ppm_max", type=float, default=None, help="Maximum ppm to display")
    args = ap.parse_args()

    spectra = [load_spectrum(p) for p in args.spectra]
    labels = (
        args.labels
        if args.labels and len(args.labels) == len(args.spectra)
        else [pathlib.Path(p).stem for p in args.spectra]
    )
    ppm_range = (args.ppm_min, args.ppm_max) if args.ppm_min or args.ppm_max else None
    out_path = pathlib.Path(args.output)

    if args.stacked:
        plot_stacked(spectra, labels, args.title, out_path, ppm_range)
    else:
        plot_overlay(spectra, labels, args.title, out_path, ppm_range)
    print(f"Plot saved → {out_path}")


if __name__ == "__main__":
    main()
