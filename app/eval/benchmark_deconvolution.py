#!/usr/bin/env python3
"""
Benchmark script for NMR deconvolution/quantification accuracy.

Generates synthetic mixtures with known compositions, runs deconvolution,
and reports accuracy metrics.

Usage:
    python -m app.eval.benchmark_deconvolution
    python -m app.eval.benchmark_deconvolution --quick
    python -m app.eval.benchmark_deconvolution --output results.json
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.tools_nmrbank import get_reference_by_smiles, warm_cache


# ============================================================================
# Configuration
# ============================================================================

# Test compounds with known NMRBank references
TEST_COMPOUNDS = {
    "toluene": "Cc1ccccc1",
    "benzene": "c1ccccc1",
    "chlorobenzene": "Clc1ccccc1",
    "anisole": "COc1ccccc1",
    "ethylbenzene": "CCc1ccccc1",
    "phenol": "Oc1ccccc1",
}

# Benchmark scenarios
BENCHMARK_SCENARIOS = [
    {
        "name": "binary_equal",
        "description": "Two components, equal amounts",
        "components": {"toluene": 0.5, "benzene": 0.5},
    },
    {
        "name": "binary_unequal",
        "description": "Two components, 70:30 ratio",
        "components": {"toluene": 0.7, "benzene": 0.3},
    },
    {
        "name": "binary_major_minor",
        "description": "Two components, major (90%) + minor (10%)",
        "components": {"toluene": 0.9, "benzene": 0.1},
    },
    {
        "name": "ternary_equal",
        "description": "Three components, equal amounts",
        "components": {"toluene": 0.33, "chlorobenzene": 0.33, "benzene": 0.34},
    },
    {
        "name": "ternary_gradient",
        "description": "Three components, 50:30:20",
        "components": {"toluene": 0.5, "chlorobenzene": 0.3, "benzene": 0.2},
    },
    {
        "name": "trace_detection",
        "description": "Major component with trace (5%)",
        "components": {"toluene": 0.95, "benzene": 0.05},
    },
]

QUICK_SCENARIOS = ["binary_equal", "ternary_equal"]


# ============================================================================
# Synthetic Spectrum Generation
# ============================================================================

def lorentzian(x: np.ndarray, center: float, intensity: float, width: float = 0.02) -> np.ndarray:
    """Generate Lorentzian peak shape."""
    return intensity * (width**2) / ((x - center)**2 + width**2)


def generate_synthetic_mixture(
    components: Dict[str, float],
    ppm_min: float = 0.0,
    ppm_max: float = 12.0,
    resolution: float = 0.001,
    linewidth: float = 0.02,
    noise_sigma: float = 0.001,
) -> Tuple[List[float], List[float], Dict]:
    """
    Generate a synthetic NMR mixture spectrum from NMRBank references.

    Args:
        components: Dict of {compound_name: weight}
        ppm_min/max: PPM range
        resolution: Points per ppm
        linewidth: Peak FWHM
        noise_sigma: Gaussian noise std

    Returns:
        (ppm_list, intensity_list, metadata)
    """
    ppm_grid = np.arange(ppm_min, ppm_max, resolution)
    mix_spectrum = np.zeros_like(ppm_grid)

    refs_used = {}

    for name, weight in components.items():
        smiles = TEST_COMPOUNDS.get(name, name)
        ref = get_reference_by_smiles(smiles)

        if ref is None:
            print(f"Warning: No reference found for {name} ({smiles})")
            continue

        refs_used[name] = {
            "smiles": smiles,
            "n_peaks": len(ref.get("ppm", [])),
        }

        # Generate continuous spectrum from peaks
        for ppm_val, inten_val in zip(ref.get("ppm", []), ref.get("intensity", [])):
            mix_spectrum += weight * lorentzian(ppm_grid, ppm_val, inten_val, linewidth)

    # Normalize to max = 1
    if np.max(mix_spectrum) > 0:
        mix_spectrum = mix_spectrum / np.max(mix_spectrum)

    # Add noise
    mix_spectrum += np.random.normal(0.0, noise_sigma, size=mix_spectrum.shape)
    mix_spectrum = np.maximum(mix_spectrum, 0.0)

    metadata = {
        "components": components,
        "refs_used": refs_used,
        "params": {
            "ppm_range": [ppm_min, ppm_max],
            "resolution": resolution,
            "linewidth": linewidth,
            "noise_sigma": noise_sigma,
        }
    }

    return ppm_grid.tolist(), mix_spectrum.tolist(), metadata


def get_reference_library(component_names: List[str]) -> List[Dict]:
    """Build reference library for deconvolution."""
    library = []
    for name in component_names:
        smiles = TEST_COMPOUNDS.get(name, name)
        ref = get_reference_by_smiles(smiles)
        if ref:
            library.append({
                "name": name,
                "smiles": smiles,
                "ppm": ref.get("ppm", []),
                "intensity": ref.get("intensity", []),
                "protons": len(ref.get("ppm", [])),
            })
    return library


# ============================================================================
# Metrics Calculation
# ============================================================================

@dataclass
class DeconvolutionMetrics:
    """Metrics for a single deconvolution run."""
    scenario_name: str
    ground_truth: Dict[str, float]
    predicted: Dict[str, float]
    mae: float
    rmse: float
    max_error: float
    r_squared: float
    detection_tp: int
    detection_fp: int
    detection_fn: int
    runtime_seconds: float
    backend: str

    def to_dict(self) -> Dict:
        return asdict(self)


def calculate_metrics(
    ground_truth: Dict[str, float],
    predicted: Dict[str, float],
    detection_threshold: float = 0.01,
) -> Tuple[float, float, float, float, int, int, int]:
    """
    Calculate deconvolution accuracy metrics.

    Returns:
        (MAE, RMSE, max_error, R², TP, FP, FN)
    """
    all_components = set(ground_truth.keys()) | set(predicted.keys())

    gt_vals = []
    pred_vals = []

    for comp in all_components:
        gt_vals.append(ground_truth.get(comp, 0.0))
        pred_vals.append(predicted.get(comp, 0.0))

    gt_arr = np.array(gt_vals)
    pred_arr = np.array(pred_vals)

    # Basic metrics
    errors = np.abs(pred_arr - gt_arr)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_error = float(np.max(errors))

    # R² (coefficient of determination)
    ss_res = np.sum((gt_arr - pred_arr)**2)
    ss_tot = np.sum((gt_arr - np.mean(gt_arr))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Detection metrics (component present vs absent)
    gt_present = set(k for k, v in ground_truth.items() if v >= detection_threshold)
    pred_present = set(k for k, v in predicted.items() if v >= detection_threshold)

    tp = len(gt_present & pred_present)
    fp = len(pred_present - gt_present)
    fn = len(gt_present - pred_present)

    return mae, rmse, max_error, r_squared, tp, fp, fn


# ============================================================================
# Benchmark Runners
# ============================================================================

def run_magnetstein_deconvolution(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    library: List[Dict],
) -> Tuple[Dict[str, float], float]:
    """Run deconvolution using Magnetstein backend."""
    import time
    from app.tools_magnetstein import quantify_single

    start = time.time()
    try:
        result = quantify_single(
            mixture_ppm=mixture_ppm,
            mixture_intensity=mixture_intensity,
            library=library,
            min_peaks=1,
        )
        concentrations = result.get("concentrations", {})
    except Exception as e:
        print(f"Magnetstein error: {e}")
        concentrations = {}

    elapsed = time.time() - start
    return concentrations, elapsed


def run_masserstein_deconvolution(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    library: List[Dict],
) -> Tuple[Dict[str, float], float]:
    """Run deconvolution using Masserstein+Gurobi backend."""
    import time
    from app.tools_deconvolve import deconvolve_spectra

    start = time.time()
    try:
        result = deconvolve_spectra(
            mixture_ppm=mixture_ppm,
            mixture_intensity=mixture_intensity,
            refs=library,
        )
        concentrations = result.get("concentrations", {})
    except Exception as e:
        print(f"Masserstein error: {e}")
        concentrations = {}

    elapsed = time.time() - start
    return concentrations, elapsed


def run_single_scenario(
    scenario: Dict,
    backends: List[str] = ["magnetstein"],
    noise_levels: List[float] = [0.001],
) -> List[DeconvolutionMetrics]:
    """Run a single benchmark scenario with specified backends and noise levels."""
    results = []

    for noise_sigma in noise_levels:
        # Generate synthetic mixture
        ppm, intensity, meta = generate_synthetic_mixture(
            components=scenario["components"],
            noise_sigma=noise_sigma,
        )

        # Build reference library
        library = get_reference_library(list(scenario["components"].keys()))

        if not library:
            print(f"Skipping {scenario['name']}: no references available")
            continue

        # Normalize ground truth to sum to 1
        gt = scenario["components"].copy()
        gt_sum = sum(gt.values())
        gt = {k: v/gt_sum for k, v in gt.items()}

        for backend in backends:
            if backend == "magnetstein":
                predicted, runtime = run_magnetstein_deconvolution(ppm, intensity, library)
            elif backend == "masserstein":
                predicted, runtime = run_masserstein_deconvolution(ppm, intensity, library)
            else:
                continue

            # Normalize predictions
            pred_sum = sum(predicted.values())
            if pred_sum > 0:
                predicted = {k: v/pred_sum for k, v in predicted.items()}

            # Calculate metrics
            mae, rmse, max_err, r2, tp, fp, fn = calculate_metrics(gt, predicted)

            metrics = DeconvolutionMetrics(
                scenario_name=f"{scenario['name']}_noise{noise_sigma}",
                ground_truth=gt,
                predicted=predicted,
                mae=mae,
                rmse=rmse,
                max_error=max_err,
                r_squared=r2,
                detection_tp=tp,
                detection_fp=fp,
                detection_fn=fn,
                runtime_seconds=runtime,
                backend=backend,
            )
            results.append(metrics)

            print(f"  [{backend}] MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")

    return results


# ============================================================================
# Main Benchmark
# ============================================================================

def run_full_benchmark(
    quick: bool = False,
    backends: List[str] = ["magnetstein"],
    output_path: Optional[str] = None,
) -> Dict:
    """
    Run the complete deconvolution benchmark suite.

    Args:
        quick: If True, run only a subset of scenarios
        backends: List of backends to test
        output_path: Optional path to save JSON results

    Returns:
        Dict with benchmark results
    """
    print("=" * 60)
    print("MixSense Deconvolution Benchmark")
    print("=" * 60)

    # Warm NMRBank cache
    print("\nLoading NMRBank references...")
    try:
        n_compounds = warm_cache()
        print(f"Loaded {n_compounds} compounds")
    except Exception as e:
        print(f"Warning: Could not load NMRBank: {e}")
        print("Using fallback test mode...")

    # Select scenarios
    if quick:
        scenarios = [s for s in BENCHMARK_SCENARIOS if s["name"] in QUICK_SCENARIOS]
        noise_levels = [0.001]
    else:
        scenarios = BENCHMARK_SCENARIOS
        noise_levels = [0.001, 0.005, 0.01]

    print(f"\nRunning {len(scenarios)} scenarios with backends: {backends}")
    print(f"Noise levels: {noise_levels}")

    all_results = []

    for scenario in scenarios:
        print(f"\n--- {scenario['name']}: {scenario['description']} ---")
        results = run_single_scenario(
            scenario,
            backends=backends,
            noise_levels=noise_levels,
        )
        all_results.extend(results)

    # Aggregate statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for backend in backends:
        backend_results = [r for r in all_results if r.backend == backend]
        if not backend_results:
            continue

        maes = [r.mae for r in backend_results]
        rmses = [r.rmse for r in backend_results]
        r2s = [r.r_squared for r in backend_results]

        print(f"\n{backend.upper()}:")
        print(f"  MAE:  mean={np.mean(maes):.4f}, std={np.std(maes):.4f}, max={np.max(maes):.4f}")
        print(f"  RMSE: mean={np.mean(rmses):.4f}, std={np.std(rmses):.4f}, max={np.max(rmses):.4f}")
        print(f"  R²:   mean={np.mean(r2s):.4f}, std={np.std(r2s):.4f}, min={np.min(r2s):.4f}")

    # Prepare output
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "quick": quick,
            "backends": backends,
            "noise_levels": noise_levels,
            "n_scenarios": len(scenarios),
        },
        "results": [r.to_dict() for r in all_results],
        "summary": {},
    }

    for backend in backends:
        backend_results = [r for r in all_results if r.backend == backend]
        if backend_results:
            output["summary"][backend] = {
                "mae_mean": float(np.mean([r.mae for r in backend_results])),
                "mae_std": float(np.std([r.mae for r in backend_results])),
                "rmse_mean": float(np.mean([r.rmse for r in backend_results])),
                "r2_mean": float(np.mean([r.r_squared for r in backend_results])),
            }

    # Save results
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return output


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark NMR deconvolution accuracy")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (fewer scenarios)")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file path")
    parser.add_argument("--backends", nargs="+", default=["magnetstein"],
                        choices=["magnetstein", "masserstein"],
                        help="Deconvolution backends to test")

    args = parser.parse_args()

    run_full_benchmark(
        quick=args.quick,
        backends=args.backends,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
