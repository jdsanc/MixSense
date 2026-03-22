#!/usr/bin/env python3
"""
Deconvolve an NMR mixture against component spectra using Masserstein + Gurobi.

Usage (CSV):
  python tool_deconvolve_nmr.py preprocessed_mix.csv preprocessed_comp0.csv preprocessed_comp1.csv \
      --protons 16 12 --names Pinene "Benzyl benzoate"

Usage (Mnova TSV export):
  python tool_deconvolve_nmr.py mix.tsv comp0.tsv comp1.tsv --protons 16 12 --mnova
"""

import os, sys, argparse, pathlib, json
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
    ap.add_argument("--threads", type=int, default=8, help="Solver/BLAS threads (default: 8)")
    ap.add_argument("--time-limit", type=int, default=300, help="Gurobi time limit seconds (default: 300)")
    ap.add_argument("--method", choices=["auto","primal","dual","barrier"], default="dual",
                    help="Gurobi LP method (default: dual)")
    ap.add_argument("--presolve", type=int, choices=[0,1,2], default=2, help="Presolve level (default: 2)")
    ap.add_argument("--numeric-focus", type=int, choices=[0,1,2,3], default=1, help="NumericFocus (default: 1)")
    ap.add_argument("--skip-crossover", action="store_true",
                    help="If using barrier, skip crossover (Crossover=0)")
    ap.add_argument("--license-file", help="Path to gurobi.lic (optional; otherwise use environment)")
    ap.add_argument("--mnova", action="store_true", help="Treat inputs as Mnova TSV (delimiter='\\t')")
    ap.add_argument("--json", action="store_true", help="Print JSON result line as well")
    ap.add_argument("--quiet", action="store_true", help="Less solver chatter")
    args = ap.parse_args()

    set_thread_env(args.threads)

    if args.license_file:
        os.environ["GRB_LICENSE_FILE"] = args.license_file

    # Gurobi sanity check
    try:
        import gurobipy as gp
        gp.setParam("LogToConsole", 0)
        _m = gp.Model()
        _m.Params.LogToConsole = 0
    except Exception:
        print("ERROR: Gurobi not ready. Make sure your license is configured.", file=sys.stderr)
        raise

    # Masserstein import
    try:
        from magnetstein.masserstein import NMRSpectrum, estimate_proportions
    except ImportError:
        print("ERROR: Could not import 'masserstein'. Install it in this environment.", file=sys.stderr)
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

    # Solver setup
    from pulp import GUROBI
    method_map = {"auto":0, "primal":1, "dual":2, "barrier":3}
    solver_kwargs = dict(
        msg=(not args.quiet),
        Threads=args.threads,
        TimeLimit=args.time_limit,
        Presolve=args.presolve,
        NumericFocus=args.numeric_focus,
    )
    if args.method != "auto":
        solver_kwargs["Method"] = method_map[args.method]
    if args.method == "barrier" and args.skip_crossover:
        solver_kwargs["Crossover"] = 0

    solver = GUROBI(**solver_kwargs)

    # Solve
    result = estimate_proportions(
        mix, spectra,
        MTD=args.kappa_mix, MTD_th=args.kappa_comp,
        verbose=(not args.quiet),
        solver=solver
    )

    props = [float(p) for p in result.get("proportions", [])]
    wd = float(result.get("Wasserstein distance", np.nan))

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
