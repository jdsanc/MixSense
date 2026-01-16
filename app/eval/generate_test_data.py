#!/usr/bin/env python3
"""
Generate synthetic test data for MixSense benchmarks.

Creates:
1. Synthetic NMR mixture spectra with known compositions
2. Ground truth JSON files
3. Dummy/mock spectra for unit tests (no NMRBank dependency)

Usage:
    python -m app.eval.generate_test_data
    python -m app.eval.generate_test_data --output-dir ./test_data
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================================
# Dummy NMR Reference Data (No NMRBank Dependency)
# ============================================================================

# Approximate peak positions for common compounds
# These are simplified/idealized for testing purposes
DUMMY_REFERENCES = {
    "benzene": {
        "smiles": "c1ccccc1",
        "ppm": [7.36],
        "intensity": [6.0],
        "protons": 6,
    },
    "toluene": {
        "smiles": "Cc1ccccc1",
        "ppm": [7.20, 7.15, 2.35],
        "intensity": [2.0, 3.0, 3.0],
        "protons": 8,
    },
    "chlorobenzene": {
        "smiles": "Clc1ccccc1",
        "ppm": [7.35, 7.25],
        "intensity": [2.0, 3.0],
        "protons": 5,
    },
    "anisole": {
        "smiles": "COc1ccccc1",
        "ppm": [7.28, 6.92, 3.80],
        "intensity": [2.0, 3.0, 3.0],
        "protons": 8,
    },
    "p_bromoanisole": {
        "smiles": "COc1ccc(Br)cc1",
        "ppm": [7.38, 6.78, 3.78],
        "intensity": [2.0, 2.0, 3.0],
        "protons": 7,
    },
    "ethylbenzene": {
        "smiles": "CCc1ccccc1",
        "ppm": [7.25, 7.18, 2.65, 1.24],
        "intensity": [2.0, 3.0, 2.0, 3.0],
        "protons": 10,
    },
    "acetone": {
        "smiles": "CC(=O)C",
        "ppm": [2.17],
        "intensity": [6.0],
        "protons": 6,
    },
    "ethanol": {
        "smiles": "CCO",
        "ppm": [3.72, 1.19],
        "intensity": [2.0, 3.0],
        "protons": 5,
    },
}


# ============================================================================
# Spectrum Generation
# ============================================================================

def lorentzian(x: np.ndarray, center: float, intensity: float, width: float) -> np.ndarray:
    """Generate Lorentzian peak shape."""
    return intensity * (width**2) / ((x - center)**2 + width**2)


def gaussian(x: np.ndarray, center: float, intensity: float, width: float) -> np.ndarray:
    """Generate Gaussian peak shape."""
    return intensity * np.exp(-0.5 * ((x - center) / width)**2)


def generate_spectrum_from_peaks(
    peak_positions: List[float],
    peak_intensities: List[float],
    ppm_min: float = 0.0,
    ppm_max: float = 12.0,
    resolution: float = 0.01,
    linewidth: float = 0.02,
    peak_shape: str = "lorentzian",
    noise_sigma: float = 0.0,
) -> tuple:
    """
    Generate a continuous spectrum from peak list.

    Returns:
        (ppm_array, intensity_array)
    """
    ppm_grid = np.arange(ppm_min, ppm_max, resolution)
    spectrum = np.zeros_like(ppm_grid)

    shape_fn = lorentzian if peak_shape == "lorentzian" else gaussian

    for pos, inten in zip(peak_positions, peak_intensities):
        if ppm_min <= pos <= ppm_max:
            spectrum += shape_fn(ppm_grid, pos, inten, linewidth)

    # Normalize
    if np.max(spectrum) > 0:
        spectrum = spectrum / np.max(spectrum)

    # Add noise
    if noise_sigma > 0:
        spectrum += np.random.normal(0, noise_sigma, size=spectrum.shape)
        spectrum = np.maximum(spectrum, 0)

    return ppm_grid, spectrum


def generate_mixture_spectrum(
    components: Dict[str, float],
    references: Dict[str, Dict] = None,
    **kwargs
) -> tuple:
    """
    Generate mixture spectrum from component weights.

    Args:
        components: {compound_name: weight}
        references: Reference spectra dict (uses DUMMY_REFERENCES if None)
        **kwargs: Passed to generate_spectrum_from_peaks

    Returns:
        (ppm_array, intensity_array, metadata)
    """
    if references is None:
        references = DUMMY_REFERENCES

    # Collect all peaks with weights
    all_peaks = []
    all_intensities = []
    refs_used = {}

    for name, weight in components.items():
        if name not in references:
            print(f"Warning: {name} not in references, skipping")
            continue

        ref = references[name]
        refs_used[name] = {"smiles": ref.get("smiles", ""), "weight": weight}

        for ppm, inten in zip(ref["ppm"], ref["intensity"]):
            all_peaks.append(ppm)
            all_intensities.append(inten * weight)

    ppm_grid, spectrum = generate_spectrum_from_peaks(
        all_peaks, all_intensities, **kwargs
    )

    metadata = {
        "components": components,
        "refs_used": refs_used,
        "generation_time": datetime.now().isoformat(),
    }

    return ppm_grid, spectrum, metadata


# ============================================================================
# Test Dataset Definitions
# ============================================================================

TEST_DATASETS = [
    # Binary mixtures
    {
        "name": "binary_equal",
        "description": "Two components, equal amounts (50:50)",
        "components": {"toluene": 0.5, "benzene": 0.5},
        "params": {"noise_sigma": 0.001, "resolution": 0.01},
    },
    {
        "name": "binary_unequal",
        "description": "Two components, unequal (70:30)",
        "components": {"toluene": 0.7, "benzene": 0.3},
        "params": {"noise_sigma": 0.001, "resolution": 0.01},
    },
    {
        "name": "binary_trace",
        "description": "Major + trace component (95:5)",
        "components": {"toluene": 0.95, "benzene": 0.05},
        "params": {"noise_sigma": 0.001, "resolution": 0.01},
    },

    # Ternary mixtures
    {
        "name": "ternary_equal",
        "description": "Three components, roughly equal",
        "components": {"toluene": 0.34, "chlorobenzene": 0.33, "benzene": 0.33},
        "params": {"noise_sigma": 0.001, "resolution": 0.01},
    },
    {
        "name": "ternary_gradient",
        "description": "Three components, gradient (50:30:20)",
        "components": {"toluene": 0.5, "chlorobenzene": 0.3, "benzene": 0.2},
        "params": {"noise_sigma": 0.001, "resolution": 0.01},
    },

    # Reaction mixture (anisole bromination)
    {
        "name": "bromination_partial",
        "description": "Partial anisole bromination (40% conversion)",
        "components": {"anisole": 0.6, "p_bromoanisole": 0.4},
        "params": {"noise_sigma": 0.001, "resolution": 0.01},
    },
    {
        "name": "bromination_complete",
        "description": "Near-complete anisole bromination (90% conversion)",
        "components": {"anisole": 0.1, "p_bromoanisole": 0.9},
        "params": {"noise_sigma": 0.001, "resolution": 0.01},
    },

    # Noise robustness tests
    {
        "name": "binary_noisy_low",
        "description": "Binary mixture with low noise",
        "components": {"toluene": 0.6, "benzene": 0.4},
        "params": {"noise_sigma": 0.005, "resolution": 0.01},
    },
    {
        "name": "binary_noisy_high",
        "description": "Binary mixture with high noise",
        "components": {"toluene": 0.6, "benzene": 0.4},
        "params": {"noise_sigma": 0.02, "resolution": 0.01},
    },

    # Resolution tests
    {
        "name": "binary_lowres",
        "description": "Binary mixture, low resolution",
        "components": {"toluene": 0.6, "benzene": 0.4},
        "params": {"noise_sigma": 0.001, "resolution": 0.05},
    },
    {
        "name": "binary_highres",
        "description": "Binary mixture, high resolution",
        "components": {"toluene": 0.6, "benzene": 0.4},
        "params": {"noise_sigma": 0.001, "resolution": 0.001},
    },
]


# ============================================================================
# File Generation
# ============================================================================

def save_spectrum_csv(path: str, ppm: np.ndarray, intensity: np.ndarray):
    """Save spectrum as CSV file."""
    with open(path, 'w') as f:
        f.write("ppm,intensity\n")
        for p, i in zip(ppm, intensity):
            f.write(f"{p:.6f},{i:.6f}\n")


def save_ground_truth(path: str, dataset: Dict, metadata: Dict):
    """Save ground truth JSON file."""
    gt = {
        "name": dataset["name"],
        "description": dataset["description"],
        "components": [
            {"name": name, "fraction": frac, "smiles": DUMMY_REFERENCES.get(name, {}).get("smiles", "")}
            for name, frac in dataset["components"].items()
        ],
        "generation_params": dataset.get("params", {}),
        "metadata": metadata,
    }
    with open(path, 'w') as f:
        json.dump(gt, f, indent=2)


def generate_all_test_data(output_dir: str):
    """Generate all test datasets."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "spectra"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "ground_truth"), exist_ok=True)

    print(f"Generating test data in: {output_dir}")

    manifest = {
        "generated": datetime.now().isoformat(),
        "datasets": [],
    }

    for dataset in TEST_DATASETS:
        name = dataset["name"]
        print(f"  Generating: {name}")

        # Generate spectrum
        ppm, intensity, metadata = generate_mixture_spectrum(
            components=dataset["components"],
            references=DUMMY_REFERENCES,
            **dataset.get("params", {})
        )

        # Save files
        spectrum_path = os.path.join(output_dir, "spectra", f"{name}.csv")
        gt_path = os.path.join(output_dir, "ground_truth", f"{name}.json")

        save_spectrum_csv(spectrum_path, ppm, intensity)
        save_ground_truth(gt_path, dataset, metadata)

        manifest["datasets"].append({
            "name": name,
            "spectrum_file": f"spectra/{name}.csv",
            "ground_truth_file": f"ground_truth/{name}.json",
            "n_points": len(ppm),
        })

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nGenerated {len(TEST_DATASETS)} datasets")
    print(f"Manifest saved to: {manifest_path}")


def generate_reference_library_json(output_path: str):
    """Generate a JSON file with all dummy references for testing."""
    refs = []
    for name, data in DUMMY_REFERENCES.items():
        refs.append({
            "name": name,
            "smiles": data["smiles"],
            "ppm": data["ppm"],
            "intensity": data["intensity"],
            "protons": data.get("protons", len(data["ppm"])),
        })

    with open(output_path, 'w') as f:
        json.dump(refs, f, indent=2)

    print(f"Reference library saved to: {output_path}")


# ============================================================================
# Time Series Generation
# ============================================================================

def generate_timeseries(
    output_dir: str,
    times: List[float],
    start_composition: Dict[str, float],
    end_composition: Dict[str, float],
    **kwargs
):
    """
    Generate time-series spectra with linear interpolation of compositions.
    """
    os.makedirs(output_dir, exist_ok=True)

    all_species = set(start_composition.keys()) | set(end_composition.keys())
    t_min, t_max = min(times), max(times)

    gt_entries = []

    for t in times:
        # Linear interpolation
        alpha = (t - t_min) / (t_max - t_min + 1e-9)
        comp = {}
        for species in all_species:
            s0 = start_composition.get(species, 0.0)
            s1 = end_composition.get(species, 0.0)
            comp[species] = (1 - alpha) * s0 + alpha * s1

        # Generate spectrum
        ppm, intensity, metadata = generate_mixture_spectrum(comp, **kwargs)

        # Save
        filename = f"t{int(t)}.csv"
        save_spectrum_csv(os.path.join(output_dir, filename), ppm, intensity)

        gt_entries.append({
            "time": t,
            "file": filename,
            "composition": comp,
        })

    # Save ground truth
    gt = {
        "type": "timeseries",
        "times": times,
        "species": list(all_species),
        "entries": gt_entries,
    }
    with open(os.path.join(output_dir, "ground_truth.json"), 'w') as f:
        json.dump(gt, f, indent=2)

    print(f"Generated time series with {len(times)} points in: {output_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate test data for MixSense benchmarks")
    parser.add_argument("--output-dir", "-o", type=str,
                        default="app/eval/data/synthetic",
                        help="Output directory for test data")

    args = parser.parse_args()

    # Generate main test datasets
    generate_all_test_data(args.output_dir)

    # Generate reference library JSON
    generate_reference_library_json(
        os.path.join(args.output_dir, "reference_library.json")
    )

    # Generate a time series example
    generate_timeseries(
        output_dir=os.path.join(args.output_dir, "timeseries_reaction"),
        times=[0, 5, 10, 15, 20, 30],
        start_composition={"anisole": 1.0, "p_bromoanisole": 0.0},
        end_composition={"anisole": 0.1, "p_bromoanisole": 0.9},
        noise_sigma=0.001,
        resolution=0.01,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
