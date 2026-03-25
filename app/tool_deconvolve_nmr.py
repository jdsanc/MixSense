#!/usr/bin/env python3
"""
Deconvolve an NMR mixture against component spectra using Magnetstein + CBC solver.

Usage (CSV):
  python tool_deconvolve_nmr.py preprocessed_mix.csv preprocessed_comp0.csv preprocessed_comp1.csv \
      --protons 16 12 --names Pinene "Benzyl benzoate"

Usage (Mnova TSV export):
  python tool_deconvolve_nmr.py mix.tsv comp0.tsv comp1.tsv --protons 16 12 --mnova
"""

import os, sys, argparse, pathlib, json; sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "magnetstein")))  # magnetstein/masserstein must shadow PyPI masserstein
import numpy as np

def set_thread_env(n):
    """Keep BLAS from oversubscribing threads."""
    for k in ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(k, str(n))

def detect_delim(path: str, default=","):
    """Heuristic: .tsv -> '\t'; else sniff first line for a tab; else comma."""
    p = pathlib.Path(path)
    if p.suffix.lower() == ".tsv":
        return "\t"
    try:
        with open(path, "r", errors="ignore") as f:
            line = f.readline()
        return "\t" if ("\t" in line) else default
    except Exception:
        return default

def load_xy(path, delimiter=None, mnova=False):
    """Load two-column spectrum file (ppm, intensity)."""
    if delimiter is None:
        delimiter = "\t" if mnova else detect_delim(path, default="," if not mnova else "\t")
    arr = np.loadtxt(path, delimiter=delimiter, usecols=[0,1])
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"{path}: expected two numeric columns (ppm,intensity)")
    return arr

def baseline_correct(arr: np.ndarray) -> np.ndarray:
    """
    Shift all intensities so the minimum value becomes 0.

    This is the minimal transformation needed to make digitized spectra
    (which may have a non-zero baseline due to the digitizer's zero-line offset)
    fully non-negative before deconvolution.
    """
    corrected = arr.copy()
    corrected[:, 1] = arr[:, 1] - arr[:, 1].min()
    return corrected

def _save_plot(out_path: str, mix_arr: np.ndarray, comp_arrays: list,
               names: list, props: list, wd: float) -> None:
    """
    Save a four-panel deconvolution plot:
      - Top:    mixture spectrum
      - Middle: each component scaled by its estimated proportion
      - Bottom: reconstructed fit overlaid on mixture, plus residual
    All spectra are interpolated to the mixture's ppm grid for the fit panels.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not available; skipping plot.", file=sys.stderr)
        return

    mix_ppm = mix_arr[:, 0]
    mix_int = mix_arr[:, 1]
    order = np.argsort(mix_ppm)
    mix_ppm, mix_int = mix_ppm[order], mix_int[order]

    n_comp = len(comp_arrays)
    colors = [f"C{i+1}" for i in range(n_comp)]

    # Interpolate each component onto the mixture ppm grid
    comp_on_grid = []
    for arr in comp_arrays:
        p, i = arr[:, 0], arr[:, 1]
        o = np.argsort(p)
        comp_on_grid.append(np.interp(mix_ppm, p[o], i[o], left=0.0, right=0.0))

    # Reconstruct: scale each component by its proportion × peak ratio
    mix_max = mix_int.max() if mix_int.max() != 0 else 1.0
    scaled = []
    for comp_int, prop in zip(comp_on_grid, props):
        comp_max = comp_int.max() if comp_int.max() != 0 else 1.0
        scaled.append(comp_int * (prop * mix_max / comp_max))

    fit = sum(scaled)
    residual = mix_int - fit

    n_rows = 2 + n_comp
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 3 * n_rows), sharex=True)

    # Row 0: mixture
    axes[0].plot(mix_ppm, mix_int, color="black", lw=0.9, label="Mixture")
    axes[0].axhline(0, color="gray", lw=0.4)
    axes[0].set_ylabel("Intensity")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[0].set_title("NMR Mixture Deconvolution")

    # Rows 1..n_comp: scaled components
    for i, (name, sc, color) in enumerate(zip(names, scaled, colors)):
        ax = axes[1 + i]
        ax.plot(mix_ppm, sc, color=color, lw=0.9,
                label=f"{name}  ({props[i]*100:.1f}%)")
        ax.axhline(0, color="gray", lw=0.4)
        ax.set_ylabel("Intensity")
        ax.legend(loc="upper left", fontsize=8)

    # Last row: fit + residual
    ax_fit = axes[-1]
    ax_fit.plot(mix_ppm, mix_int, color="black", lw=0.9, alpha=0.5, label="Mixture")
    ax_fit.plot(mix_ppm, fit, color="red", lw=0.9, linestyle="--", label="Fit (sum)")
    ax_fit.fill_between(mix_ppm, residual, 0, color="gray", alpha=0.3, label="Residual")
    ax_fit.axhline(0, color="gray", lw=0.4)
    ax_fit.set_ylabel("Intensity")
    ax_fit.set_xlabel("Chemical shift (ppm)")
    ax_fit.legend(loc="upper left", fontsize=8)
    ax_fit.set_title(f"Fit vs Mixture  (Wasserstein distance = {wd:.5f})")

    axes[-1].invert_xaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Estimate component proportions in an NMR mixture.")
    ap.add_argument("mixture", help="CSV/TSV file with mixture: columns [ppm, intensity]")
    ap.add_argument("components", nargs="+", help="CSV/TSV files for components: columns [ppm, intensity]")
    ap.add_argument("--protons", type=int, nargs="+", help="Proton counts for each component (e.g. 16 12)")
    ap.add_argument("--names", nargs="+", help='Names for components (e.g. Pinene "Benzyl benzoate")')
    ap.add_argument("--kappa-mix", type=float, default=0.25, help="Kappa for mixture (default: 0.25)")
    ap.add_argument("--kappa-comp", type=float, default=0.22, help="Kappa for components (default: 0.22)")
    ap.add_argument("--threads", type=int, default=8, help="BLAS threads (default: 8)")
    ap.add_argument("--mnova", action="store_true", help="Treat inputs as Mnova TSV (delimiter='\\t')")
    ap.add_argument("--baseline-correct", action="store_true",
                    help="Shift each spectrum so its minimum intensity becomes 0. "
                         "Recommended for digitized spectra with a non-zero baseline.")
    ap.add_argument("--plot", metavar="FILE", default=None,
                    help="Save a deconvolution plot to FILE (e.g. result.png). "
                         "Shows mixture, scaled components, reconstructed fit, and residual.")
    ap.add_argument("--json", action="store_true", help="Print JSON result line as well")
    ap.add_argument("--quiet", action="store_true", help="Less solver chatter")
    args = ap.parse_args()

    set_thread_env(args.threads)

    # Magnetstein import
    try:
        from magnetstein.masserstein import NMRSpectrum, estimate_proportions
        from pulp import LpSolverDefault
    except ImportError as e:
        print(f"ERROR: Could not import magnetstein/pulp: {e}", file=sys.stderr)
        raise

    # Load data
    mix_arr = load_xy(args.mixture, mnova=args.mnova)
    comp_arrays = [load_xy(p, mnova=args.mnova) for p in args.components]

    # Optional baseline correction
    if args.baseline_correct:
        if not args.quiet:
            print("Baseline correction: shifting each spectrum so its minimum intensity = 0.")
        mix_arr = baseline_correct(mix_arr)
        comp_arrays = [baseline_correct(a) for a in comp_arrays]

    n = len(comp_arrays)
    names = args.names if args.names and len(args.names) == n else [f"comp{i}" for i in range(n)]
    if args.names and len(args.names) != n:
        print("WARNING: --names length does not match number of components; using default names.", file=sys.stderr)

    if args.protons and len(args.protons) != n:
        print("ERROR: --protons length must equal number of components.", file=sys.stderr)
        sys.exit(2)
    protons = args.protons if args.protons else [1]*n
    if not args.protons:
        print("NOTE: No --protons provided; assuming 1 for each component.", file=sys.stderr)

    # Build spectrum objects
    mix = NMRSpectrum(confs=list(zip(mix_arr[:,0], mix_arr[:,1])))
    spectra = []
    for i, arr in enumerate(comp_arrays):
        spectra.append(NMRSpectrum(confs=list(zip(arr[:,0], arr[:,1])), protons=protons[i]))

    # Normalize
    mix.trim_negative_intensities()
    mix.normalize()
    for sp in spectra:
        sp.trim_negative_intensities()
        sp.normalize()

    # Solve
    result = estimate_proportions(
        mix, spectra,
        MTD=args.kappa_mix, MTD_th=args.kappa_comp,
        verbose=(not args.quiet),
        solver=LpSolverDefault,
    )

    props = [float(p) for p in result.get("proportions", [])]
    wd = float(result.get("Wasserstein distance", float("nan")))

    # Output
    print("\nEstimated proportions:")
    width = max(len(s) for s in names) + 2
    for name, val in zip(names, props):
        print(f"  {name.ljust(width)} {val:.6f}")
    print(f"\nWasserstein distance: {wd:.12f}")

    if args.json:
        out = {"proportions": dict(zip(names, props)), "Wasserstein distance": wd}
        print("\nJSON:", json.dumps(out))

    if args.plot:
        _save_plot(args.plot, mix_arr, comp_arrays, names, props, wd)
        if not args.quiet:
            print(f"\nPlot saved → {args.plot}")

if __name__ == "__main__":
    main()
