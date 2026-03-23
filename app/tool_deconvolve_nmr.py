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

if __name__ == "__main__":
    main()
