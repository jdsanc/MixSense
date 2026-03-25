#!/usr/bin/env python3
"""
NMR reaction kinetics: supervised deconvolution at each time point → composition vs time.

Runs app/tool_deconvolve_nmr.py on a series of crude spectra recorded at known times,
collects the estimated mole fractions and Wasserstein distances, saves a CSV table
and a kinetics plot.

Usage:
    python nmr_kinetics.py \
        --refs ref_a.csv ref_b.csv \
        --timepoints t000min.csv t005min.csv t010min.csv \
        --times 0 5 10 \
        --time_unit min \
        --protons 18 18 \
        --names "borneol" "isoborneol" \
        --baseline_correct \
        --output_dir results/kinetics

Requirements:
    - Environment: mixsense (uv sync)
    - Required packages: numpy, matplotlib
    - app/tool_deconvolve_nmr.py must be accessible relative to project root
"""

import argparse
import csv
import json
import os
import pathlib
import subprocess
import sys

import numpy as np


def run_deconvolution(
    crude: str,
    refs: list,
    names: list,
    protons: list,
    baseline_correct: bool,
    kappa_mix: float,
    kappa_comp: float,
    tool_path: str,
) -> dict:
    """
    Call tool_deconvolve_nmr.py for a single time point and parse its JSON output.

    Returns:
        Dict with keys 'proportions' (dict name→float) and 'Wasserstein distance' (float).
    """
    cmd = [sys.executable, tool_path, crude] + refs + ["--json", "--quiet"]
    if names:
        cmd += ["--names"] + names
    if protons:
        cmd += ["--protons"] + [str(p) for p in protons]
    if baseline_correct:
        cmd += ["--baseline-correct"]
    cmd += ["--kappa-mix", str(kappa_mix), "--kappa-comp", str(kappa_comp)]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Deconvolution failed for {crude}:\n{result.stderr}")

    # Parse the JSON line from stdout
    for line in result.stdout.splitlines():
        if line.strip().startswith("JSON:"):
            return json.loads(line.split("JSON:", 1)[1].strip())
    raise RuntimeError(f"No JSON output found for {crude}. stdout:\n{result.stdout}")


def save_kinetics_plot(
    times: list,
    time_unit: str,
    names: list,
    proportions_over_time: list,
    wd_over_time: list,
    out_path: pathlib.Path,
) -> None:
    """
    Save a two-panel kinetics plot:
      - Top:    mole fraction vs time for each component (line plot).
      - Bottom: Wasserstein distance vs time (fit quality indicator).

    Args:
        times: List of time values.
        time_unit: Label for the time axis (e.g. 'min', 'h', 's').
        names: Component names.
        proportions_over_time: List of dicts {name: fraction} per time point.
        wd_over_time: Wasserstein distances per time point.
        out_path: Output file path (.png or .pdf).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    for i, name in enumerate(names):
        fracs = [p.get(name, float("nan")) for p in proportions_over_time]
        ax1.plot(times, [f * 100 for f in fracs], "o-", color=f"C{i}",
                 lw=1.5, markersize=5, label=name)

    ax1.set_ylabel("Mole fraction (%)")
    ax1.set_ylim(-5, 105)
    ax1.axhline(0, color="gray", lw=0.4)
    ax1.legend(loc="center right")
    ax1.set_title("NMR Reaction Kinetics — Component Composition Over Time")

    ax2.semilogy(times, wd_over_time, "s--", color="gray", lw=1.2, markersize=4)
    ax2.set_ylabel("Wasserstein\ndistance")
    ax2.set_xlabel(f"Time ({time_unit})")
    ax2.set_title("Fit quality (lower = better)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description="Run NMR deconvolution at each time point and plot kinetics."
    )
    ap.add_argument("--refs", nargs="+", required=True,
                    help="Reference spectrum files for each component")
    ap.add_argument("--timepoints", nargs="+", required=True,
                    help="Crude spectrum files ordered by time")
    ap.add_argument("--times", type=float, nargs="+", required=True,
                    help="Time values matching --timepoints (e.g. 0 5 10 20)")
    ap.add_argument("--time_unit", default="min",
                    help="Time axis label (default: min)")
    ap.add_argument("--protons", type=int, nargs="+",
                    help="Proton counts per reference component")
    ap.add_argument("--names", nargs="+",
                    help="Component names (must match number of --refs)")
    ap.add_argument("--baseline_correct", action="store_true",
                    help="Shift each spectrum so its minimum intensity = 0 before deconvolution.")
    ap.add_argument("--kappa_mix", type=float, default=0.25)
    ap.add_argument("--kappa_comp", type=float, default=0.22)
    ap.add_argument("--output_dir", default="kinetics_results",
                    help="Directory for kinetics CSV and plot")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if len(args.times) != len(args.timepoints):
        print("ERROR: --times and --timepoints must have the same length.", file=sys.stderr)
        sys.exit(1)

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate tool_deconvolve_nmr.py relative to this script or cwd
    script_dir = pathlib.Path(__file__).parent
    for candidate in [
        script_dir.parent.parent.parent / "app" / "tool_deconvolve_nmr.py",
        pathlib.Path("app") / "tool_deconvolve_nmr.py",
    ]:
        if candidate.exists():
            tool_path = str(candidate)
            break
    else:
        print("ERROR: Cannot find app/tool_deconvolve_nmr.py. Run from the MixSense project root.",
              file=sys.stderr)
        sys.exit(1)

    n_refs = len(args.refs)
    names = args.names if args.names and len(args.names) == n_refs \
        else [f"comp{i}" for i in range(n_refs)]

    proportions_over_time = []
    wd_over_time = []

    for t, tp_path in zip(args.times, args.timepoints):
        if not args.quiet:
            print(f"  t={t} {args.time_unit}  ← {os.path.basename(tp_path)}", end="  ")
        try:
            res = run_deconvolution(
                crude=tp_path, refs=args.refs, names=names, protons=args.protons,
                baseline_correct=args.baseline_correct,
                kappa_mix=args.kappa_mix, kappa_comp=args.kappa_comp,
                tool_path=tool_path,
            )
            proportions_over_time.append(res["proportions"])
            wd = res["Wasserstein distance"]
            wd_over_time.append(wd)
            if not args.quiet:
                frac_str = "  ".join(f"{n}={res['proportions'].get(n, 0)*100:.1f}%"
                                     for n in names)
                print(f"{frac_str}  WD={wd:.5f}")
        except Exception as e:
            print(f"FAILED: {e}", file=sys.stderr)
            proportions_over_time.append({n: float("nan") for n in names})
            wd_over_time.append(float("nan"))

    # Save CSV table
    csv_path = out_dir / "kinetics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([f"time_{args.time_unit}"] + names + ["wasserstein_distance"])
        for t, props, wd in zip(args.times, proportions_over_time, wd_over_time):
            writer.writerow([t] + [props.get(n, float("nan")) for n in names] + [wd])
    if not args.quiet:
        print(f"\nKinetics table → {csv_path}")

    # Save plot
    plot_path = out_dir / "kinetics_plot.png"
    save_kinetics_plot(args.times, args.time_unit, names,
                       proportions_over_time, wd_over_time, plot_path)
    if not args.quiet:
        print(f"Kinetics plot  → {plot_path}")


if __name__ == "__main__":
    main()
