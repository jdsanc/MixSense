"""
Blind source separation of NMR mixture time-series via PCA and NMF.

Stacks multiple input spectra into a matrix, uses PCA to estimate the number of
chemically distinct components, then NMF to recover pure-component spectral profiles
and their relative abundances at each time point.

Usage:
    python nmr_blind_separation.py spectrum_t0.csv spectrum_t1.csv spectrum_t2.csv \\
        --n_components 2 --output_dir nmr_separation_results --labels 0min 30min 60min

Requirements:
    - Environment: mixsense (uv sync)
    - Required packages: numpy, scikit-learn, matplotlib
"""

import argparse
import json
import pathlib
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, NMF

sys.path.insert(0, os.path.dirname(__file__))
from nmr_io import load_time_series


def estimate_n_components_pca(evr: np.ndarray, threshold: float = 0.95) -> int:
    """
    Estimate the number of components from PCA explained variance ratios.

    Args:
        evr: Explained variance ratio array from PCA.
        threshold: Cumulative variance threshold (default 0.95).

    Returns:
        Smallest integer k such that the first k components explain >= threshold of variance.
        Minimum return value is 2.
    """
    cumvar = np.cumsum(evr)
    idx = int(np.searchsorted(cumvar, threshold)) + 1
    return max(2, min(idx, len(evr)))


def run_pca(matrix: np.ndarray, max_components: int = 10) -> tuple:
    """
    Fit PCA on the spectrum matrix.

    Args:
        matrix: (n_spectra, n_points) matrix of intensity values.
        max_components: Maximum number of PCA components.

    Returns:
        Tuple of (fitted PCA model, explained_variance_ratio array).
    """
    n_comp = min(max_components, matrix.shape[0], matrix.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(matrix)
    return pca, pca.explained_variance_ratio_


def run_nmf(matrix: np.ndarray, n_components: int, random_state: int = 42) -> tuple:
    """
    Run NMF to extract pure component spectra from mixture time-series.

    Args:
        matrix: (n_spectra, n_points) intensity matrix. Negative values are clipped to 0.
        n_components: Number of pure spectral components to extract.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of:
            W: abundance matrix of shape (n_spectra, n_components).
            H: component spectra matrix of shape (n_components, n_points).
    """
    matrix_nn = np.clip(matrix, 0, None)
    nmf = NMF(n_components=n_components, init="nndsvda", random_state=random_state, max_iter=500)
    W = nmf.fit_transform(matrix_nn)
    H = nmf.components_
    return W, H


def plot_scree(evr: np.ndarray, out_path: pathlib.Path) -> None:
    """Save a PCA cumulative explained variance (scree) plot."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(evr) + 1), np.cumsum(evr) * 100, "o-", color="steelblue")
    ax.axhline(95, color="gray", linestyle="--", linewidth=0.8, label="95 % threshold")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA Scree Plot")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_components(grid: np.ndarray, H: np.ndarray, out_path: pathlib.Path) -> None:
    """Save a stacked plot of NMF pure-component spectra."""
    n = H.shape[0]
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, (ax, h) in enumerate(zip(axes, H)):
        ax.plot(grid, h, color=f"C{i}", linewidth=0.9)
        ax.set_ylabel(f"Component {i}")
        ax.invert_xaxis()
    axes[-1].set_xlabel("Chemical shift (ppm)")
    fig.suptitle("NMF — Recovered Pure Component Spectra")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_abundances(W: np.ndarray, labels: list, out_path: pathlib.Path) -> None:
    """Save a line plot of component abundances over time."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(W.shape[1]):
        ax.plot(range(len(labels)), W[:, i], "o-", label=f"Component {i}", color=f"C{i}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("NMF abundance (a.u.)")
    ax.set_title("Component Abundances Over Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Blind NMR source separation: PCA to estimate component count, "
            "NMF to recover pure spectra and abundances."
        )
    )
    ap.add_argument("spectra", nargs="+", help="Input .csv or .xy spectrum files (time-ordered)")
    ap.add_argument(
        "--n_components", type=int, default=None,
        help="Number of NMF components. If omitted, estimated from PCA (95%% variance threshold).",
    )
    ap.add_argument("--ppm_min", type=float, default=None, help="Minimum ppm for common grid")
    ap.add_argument("--ppm_max", type=float, default=None, help="Maximum ppm for common grid")
    ap.add_argument("--n_points", type=int, default=2000, help="Grid interpolation points (default: 2000)")
    ap.add_argument("--output_dir", default="nmr_separation_results", help="Output directory")
    ap.add_argument(
        "--labels", nargs="+",
        help="Time-point labels for x-axis (e.g. 0min 10min 20min). Defaults to file stems.",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"Loading {len(args.spectra)} spectra ...")
    grid, matrix = load_time_series(
        args.spectra,
        n_points=args.n_points,
        ppm_min=args.ppm_min,
        ppm_max=args.ppm_max,
    )
    labels = (
        args.labels
        if args.labels and len(args.labels) == len(args.spectra)
        else [pathlib.Path(p).stem for p in args.spectra]
    )

    # PCA
    _, evr = run_pca(matrix, max_components=min(10, len(args.spectra)))
    plot_scree(evr, out_dir / "pca_scree.png")
    if not args.quiet:
        print(f"PCA scree plot saved -> {out_dir}/pca_scree.png")

    n_components = args.n_components
    if n_components is None:
        n_components = estimate_n_components_pca(evr)
        if not args.quiet:
            print(f"Auto-detected n_components = {n_components} (95 % variance threshold)")

    # NMF
    if not args.quiet:
        print(f"Running NMF (n_components={n_components}) ...")
    W, H = run_nmf(matrix, n_components)

    # Save component spectra as CSV
    for i, h in enumerate(H):
        comp_path = out_dir / f"component_{i}.csv"
        np.savetxt(
            comp_path,
            np.column_stack([grid, h]),
            delimiter=",",
            header="ppm,intensity",
            comments="",
        )
        if not args.quiet:
            print(f"  Component {i} -> {comp_path}")

    # Save abundance matrix
    abundance_path = out_dir / "abundances.csv"
    header = "time_point," + ",".join(f"component_{i}" for i in range(n_components))
    with open(abundance_path, "w") as f:
        f.write(header + "\n")
        for label, row in zip(labels, W):
            f.write(label + "," + ",".join(f"{v:.6f}" for v in row) + "\n")
    if not args.quiet:
        print(f"Abundance matrix -> {abundance_path}")

    # Plots
    plot_components(grid, H, out_dir / "nmf_components.png")
    plot_abundances(W, labels, out_dir / "nmf_abundances.png")
    if not args.quiet:
        print(f"Plots saved -> {out_dir}/")

    # Summary JSON
    summary = {
        "n_spectra": len(args.spectra),
        "n_components": n_components,
        "pca_explained_variance_ratio": evr.tolist(),
        "output_dir": str(out_dir),
        "component_files": [f"component_{i}.csv" for i in range(n_components)],
        "abundance_file": "abundances.csv",
        "plots": ["pca_scree.png", "nmf_components.png", "nmf_abundances.png"],
    }
    with open(out_dir / "separation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if not args.quiet:
        print(f"Summary -> {out_dir}/separation_summary.json")


if __name__ == "__main__":
    main()
