# usage: python -m app.examples.generate_demo_csv
"""
Generate dense continuous NMR spectra for demo purposes.
Creates spectra compatible with magnetstein's deconvolution algorithms.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from tools_nmrbank import get_reference_by_smiles

# Default species (change if you prefer others)
TOLUENE = "Cc1ccccc1"                      # toluene
CHLOROBENZENE = "Clc1ccccc1"               # chlorobenzene
BENZENE = "c1ccccc1"                       # benzene


def lorentzian(x: np.ndarray, center: float, intensity: float, width: float = 0.02) -> np.ndarray:
    """
    Generate a Lorentzian peak shape (typical NMR lineshape).
    
    Args:
        x: ppm grid
        center: peak center position (ppm)
        intensity: peak height/area
        width: half-width at half-maximum (ppm), default 0.02 ppm
    
    Returns:
        Lorentzian peak values on the grid
    """
    return intensity * (width**2) / ((x - center)**2 + width**2)


def gaussian(x: np.ndarray, center: float, intensity: float, width: float = 0.02) -> np.ndarray:
    """
    Generate a Gaussian peak shape.
    
    Args:
        x: ppm grid
        center: peak center position (ppm)
        intensity: peak height/area
        width: standard deviation (ppm), default 0.02 ppm
    
    Returns:
        Gaussian peak values on the grid
    """
    return intensity * np.exp(-0.5 * ((x - center) / width)**2)


def peaks_to_spectrum(
    ppm_grid: np.ndarray,
    peak_positions: List[float],
    peak_intensities: List[float],
    peak_shape: str = "lorentzian",
    linewidth: float = 0.02
) -> np.ndarray:
    """
    Convert peak list to continuous spectrum.
    
    Args:
        ppm_grid: dense ppm axis
        peak_positions: list of peak centers (ppm)
        peak_intensities: list of peak heights/integrals
        peak_shape: "lorentzian" or "gaussian"
        linewidth: peak width parameter
    
    Returns:
        Continuous spectrum array
    """
    spectrum = np.zeros_like(ppm_grid)
    shape_func = lorentzian if peak_shape == "lorentzian" else gaussian
    
    for pos, inten in zip(peak_positions, peak_intensities):
        spectrum += shape_func(ppm_grid, pos, inten, linewidth)
    
    return spectrum


def make_dense_grid(ppm_min: float = 0.0, ppm_max: float = 12.0, resolution: float = 0.001) -> np.ndarray:
    """
    Create a dense ppm grid for continuous spectra.
    
    Args:
        ppm_min: minimum ppm value
        ppm_max: maximum ppm value  
        resolution: ppm spacing between points
    
    Returns:
        Dense ppm grid array
    """
    return np.arange(ppm_min, ppm_max, resolution)


def make_single(
    out_csv: str,
    weights: Dict[str, float],
    noise_sigma: float = 0.001,
    ppm_min: float = 0.0,
    ppm_max: float = 12.0,
    resolution: float = 0.001,
    linewidth: float = 0.02,
    peak_shape: str = "lorentzian"
):
    """
    Generate a single mixture spectrum CSV with dense continuous data.
    
    Args:
        out_csv: output CSV path
        weights: dict of SMILES -> coefficient (relative amount)
        noise_sigma: standard deviation of Gaussian noise to add
        ppm_min: minimum ppm value
        ppm_max: maximum ppm value
        resolution: ppm spacing between points
        linewidth: peak width parameter
        peak_shape: "lorentzian" or "gaussian"
    """
    # Load references from NMRBank
    refs = {}
    for smi in weights:
        ref = get_reference_by_smiles(smi)
        if ref is None:
            raise ValueError(f"No NMRBank reference for {smi}")
        refs[smi] = ref
        print(f"  Loaded {smi}: {len(ref['ppm'])} peaks")

    # Create dense ppm grid
    ppm_grid = make_dense_grid(ppm_min, ppm_max, resolution)
    print(f"  Created ppm grid: {len(ppm_grid)} points ({ppm_min}-{ppm_max} ppm)")

    # Build mixture spectrum
    mix = np.zeros_like(ppm_grid)
    for smi, w in weights.items():
        r = refs[smi]
        component_spectrum = peaks_to_spectrum(
            ppm_grid,
            r["ppm"],
            r["intensity"],
            peak_shape=peak_shape,
            linewidth=linewidth
        )
        mix += float(w) * component_spectrum

    # Normalize to max = 1
    if np.max(np.abs(mix)) > 0:
        mix = mix / np.max(np.abs(mix))
    
    # Add light noise
    mix = mix + np.random.normal(0.0, noise_sigma, size=mix.shape)
    
    # Clip any negative values from noise
    mix = np.maximum(mix, 0.0)

    # Save to CSV (no header for compatibility with magnetstein examples)
    df = pd.DataFrame({"ppm": ppm_grid, "intensity": mix})
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(ppm_grid)} data points)")


def make_timeseries(
    out_dir: str,
    times: List[float],
    start: Dict[str, float],
    end: Dict[str, float],
    noise_sigma: float = 0.001,
    ppm_min: float = 0.0,
    ppm_max: float = 12.0,
    resolution: float = 0.001,
    linewidth: float = 0.02,
    peak_shape: str = "lorentzian"
):
    """
    Generate time-series spectra with linear interpolation of component weights.
    
    Args:
        out_dir: output directory for CSV files
        times: list of time points
        start: initial weight dict (SMILES -> weight)
        end: final weight dict (SMILES -> weight)
        noise_sigma: standard deviation of Gaussian noise
        ppm_min: minimum ppm value
        ppm_max: maximum ppm value
        resolution: ppm spacing between points
        linewidth: peak width parameter
        peak_shape: "lorentzian" or "gaussian"
    """
    os.makedirs(out_dir, exist_ok=True)
    species = list({*start.keys(), *end.keys()})
    
    print(f"\nGenerating time-series with {len(times)} time points...")
    print(f"Species: {species}")
    
    for t in times:
        # Linear interpolation of weights
        alpha = (t - min(times)) / (max(times) - min(times) + 1e-9)
        w = {}
        for smi in species:
            s0 = float(start.get(smi, 0.0))
            s1 = float(end.get(smi, 0.0))
            w[smi] = (1 - alpha) * s0 + alpha * s1
        
        out_csv = os.path.join(out_dir, f"t{t}.csv")
        print(f"\nTime {t}:")
        for smi, weight in w.items():
            print(f"  {smi}: {weight:.3f}")
        
        make_single(
            out_csv, w,
            noise_sigma=noise_sigma,
            ppm_min=ppm_min,
            ppm_max=ppm_max,
            resolution=resolution,
            linewidth=linewidth,
            peak_shape=peak_shape
        )


if __name__ == "__main__":
    print("=" * 60)
    print("Generating dense NMR spectra for magnetstein demo")
    print("=" * 60)
    
    # Configuration for dense spectra
    config = {
        "ppm_min": 0.0,
        "ppm_max": 12.0,
        "resolution": 0.001,  # 12000 points for 0-12 ppm range
        "linewidth": 0.03,    # Typical NMR linewidth
        "noise_sigma": 0.002,
        "peak_shape": "lorentzian"
    }
    
    print(f"\nSpectrum configuration:")
    print(f"  PPM range: {config['ppm_min']} - {config['ppm_max']}")
    print(f"  Resolution: {config['resolution']} ppm")
    print(f"  Expected points: {int((config['ppm_max'] - config['ppm_min']) / config['resolution'])}")
    print(f"  Linewidth: {config['linewidth']} ppm")
    print(f"  Peak shape: {config['peak_shape']}")
    
    # Example 1: single mixture (toluene:chlorobenzene:benzene)
    print("\n" + "-" * 60)
    print("Example 1: Single mixture spectrum")
    print("-" * 60)
    out_one = os.path.join(os.path.dirname(__file__), "benzene_derivatives_demo.csv")
    make_single(
        out_one,
        {
            TOLUENE: 1.0,
            CHLOROBENZENE: 0.7,
            BENZENE: 0.1,
        },
        **config
    )

    # Example 2: time-series (toluene consumed -> chlorobenzene grows)
    print("\n" + "-" * 60)
    print("Example 2: Time-series spectra")
    print("-" * 60)
    out_dir = os.path.join(os.path.dirname(__file__), "benzene_derivatives_series")
    times = [0, 5, 10, 15, 20, 30]  # minutes
    make_timeseries(
        out_dir,
        times=times,
        start={TOLUENE: 1.0, CHLOROBENZENE: 0.0, BENZENE: 0.0},
        end={TOLUENE: 0.2, CHLOROBENZENE: 0.7, BENZENE: 0.1},
        **config
    )
    
    print("\n" + "=" * 60)
    print("Done! Generated dense spectra compatible with magnetstein.")
    print("=" * 60)
