# usage: python -m app.examples.generate_demo_csv
import os, json, math
import numpy as np
import pandas as pd

from typing import Dict, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tools_nmrbank import get_reference_by_smiles

# Default species (change if you prefer others)
TOLUENE = "Cc1ccccc1"                      # toluene
CHLOROBENZENE = "Clc1ccccc1"               # chlorobenzene
BENZENE = "c1ccccc1"                       # benzene

def _interp_to_grid(ppm_ref, inten_ref, ppm_grid):
    # Interpolate onto a common ppm grid
    return np.interp(ppm_grid, ppm_ref, inten_ref, left=0.0, right=0.0)

def make_single(out_csv: str, weights: Dict[str, float], noise_sigma=0.002):
    """
    weights: dict of SMILES -> coefficient (relative amount)
    Writes two-column CSV: ppm,intensity
    """
    refs = {}
    for smi in weights:
        ref = get_reference_by_smiles(smi)
        if ref is None:
            raise ValueError(f"No NMRBank reference for {smi}")
        refs[smi] = ref

    # Choose base ppm grid (first species)
    first = next(iter(refs.values()))
    ppm_grid = np.array(first["ppm"], dtype=float)

    mix = np.zeros_like(ppm_grid, dtype=float)
    for smi, w in weights.items():
        r = refs[smi]
        y = _interp_to_grid(np.array(r["ppm"], float), np.array(r["intensity"], float), ppm_grid)
        mix += float(w) * y

    # Normalize and add light noise
    if np.max(np.abs(mix)) > 0:
        mix = mix / np.max(np.abs(mix))
    mix = mix + np.random.normal(0.0, noise_sigma, size=mix.shape)

    df = pd.DataFrame({"ppm": ppm_grid, "intensity": mix})
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

def make_timeseries(out_dir: str, times: List[float], start: Dict[str, float], end: Dict[str, float], noise_sigma=0.002):
    """
    Linear interpolation of weights from start to end over 'times'.
    Writes files named t{time}.csv in out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    species = list({*start.keys(), *end.keys()})
    for t in times:
        alpha = (t - min(times)) / (max(times) - min(times) + 1e-9)
        w = {}
        for smi in species:
            s0 = float(start.get(smi, 0.0))
            s1 = float(end.get(smi, 0.0))
            w[smi] = (1 - alpha) * s0 + alpha * s1
        out_csv = os.path.join(out_dir, f"t{t}.csv")
        make_single(out_csv, w, noise_sigma=noise_sigma)

if __name__ == "__main__":
    # Example 1: single mixture (toluene:chlorobenzene ~ 8:1)
    out_one = os.path.join(os.path.dirname(__file__), "benzene_derivatives_demo.csv")
    make_single(out_one, {
        TOLUENE: 1.0,
        CHLOROBENZENE: 0.7,
        BENZENE: 0.1,
    })

    # Example 2: time-series (toluene consumed -> chlorobenzene grows)
    out_dir = os.path.join(os.path.dirname(__file__), "benzene_derivatives_series")
    times = [0, 5, 10, 15, 20, 30]  # minutes
    make_timeseries(
        out_dir,
        times=times,
        start={TOLUENE: 1.0, CHLOROBENZENE: 0.0, BENZENE: 0.0},
        end={TOLUENE: 0.2, CHLOROBENZENE: 0.7, BENZENE: 0.1},
    )
