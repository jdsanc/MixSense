#!/usr/bin/env python3
"""
Comprehensive benchmark comparing MixSense against baseline methods.

This script evaluates deconvolution accuracy across:
- Multiple methods (MixSense, NNLS variants, peak matching, etc.)
- Multiple difficulty levels (easy, medium, hard domain gap)
- Multiple mixture types (binary, ternary, trace detection, etc.)

Usage:
    python -m app.eval.benchmark_with_baselines
    python -m app.eval.benchmark_with_baselines --difficulty medium
    python -m app.eval.benchmark_with_baselines --output results.json
"""

import os
import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.eval.spectrum_perturbation import (
    DifficultyLevel,
    DIFFICULTY_PARAMS,
    perturb_spectrum,
    lorentzian,
)
from app.eval.baselines import (
    BASELINE_METHODS,
    run_all_baselines,
    baseline_nnls,
)


# ============================================================================
# Test Configuration
# ============================================================================

# Reference spectra (same as realistic_evaluation.py)
REFERENCE_SPECTRA = {
    "benzene": {
        "smiles": "c1ccccc1",
        "ppm": [7.36],
        "intensity": [1.0],
        "protons": 6,
    },
    "toluene": {
        "smiles": "Cc1ccccc1",
        "ppm": [7.17, 7.25, 2.34],
        "intensity": [0.6, 0.4, 1.0],
        "protons": 8,
    },
    "chlorobenzene": {
        "smiles": "Clc1ccccc1",
        "ppm": [7.35, 7.28, 7.22],
        "intensity": [0.4, 0.4, 0.2],
        "protons": 5,
    },
    "anisole": {
        "smiles": "COc1ccccc1",
        "ppm": [7.28, 6.95, 6.88, 3.80],
        "intensity": [0.4, 0.4, 0.2, 1.0],
        "protons": 8,
    },
    "p_bromoanisole": {
        "smiles": "COc1ccc(Br)cc1",
        "ppm": [7.38, 6.80, 3.78],
        "intensity": [0.5, 0.5, 1.0],
        "protons": 7,
    },
    "bromobenzene": {
        "smiles": "Brc1ccccc1",
        "ppm": [7.50, 7.30, 7.22],
        "intensity": [0.4, 0.4, 0.2],
        "protons": 5,
    },
    "ethyl_acetate": {
        "smiles": "CCOC(C)=O",
        "ppm": [4.12, 2.04, 1.26],
        "intensity": [0.67, 1.0, 1.0],
        "protons": 8,
    },
    "acetic_acid": {
        "smiles": "CC(=O)O",
        "ppm": [2.10],
        "intensity": [1.0],
        "protons": 3,
    },
    "ethanol": {
        "smiles": "CCO",
        "ppm": [3.72, 1.19],
        "intensity": [0.67, 1.0],
        "protons": 5,
    },
}

# Test cases
TEST_CASES = [
    {"id": "D1", "name": "Binary equal (50:50)", "composition": {"toluene": 0.5, "benzene": 0.5}, "category": "binary"},
    {"id": "D2", "name": "Binary unequal (70:30)", "composition": {"toluene": 0.7, "benzene": 0.3}, "category": "binary"},
    {"id": "D3", "name": "Trace detection (90:10)", "composition": {"toluene": 0.9, "benzene": 0.1}, "category": "trace"},
    {"id": "D4", "name": "Trace detection (95:5)", "composition": {"toluene": 0.95, "benzene": 0.05}, "category": "trace"},
    {"id": "D5", "name": "Ternary equal", "composition": {"toluene": 0.34, "chlorobenzene": 0.33, "benzene": 0.33}, "category": "ternary"},
    {"id": "D6", "name": "Ternary gradient", "composition": {"toluene": 0.5, "chlorobenzene": 0.3, "benzene": 0.2}, "category": "ternary"},
    {"id": "D7", "name": "Bromination partial", "composition": {"anisole": 0.6, "p_bromoanisole": 0.4}, "category": "reaction"},
    {"id": "D8", "name": "Bromination complete", "composition": {"anisole": 0.1, "p_bromoanisole": 0.9}, "category": "reaction"},
]

# Methods to compare
METHODS_TO_COMPARE = [
    "nnls",           # Our default (same as simple_deconvolve)
    "nnls_l2",        # Ridge regularization
    "nnls_l1",        # LASSO/sparse
    "peak_matching",  # Heuristic
    "integration",    # Classic NMR
    "uniform",        # Naive baseline
    "random",         # Lower bound
]


# ============================================================================
# Mixture Generation
# ============================================================================

def generate_mixture_spectrum(
    composition: Dict[str, float],
    references: Dict[str, Dict],
    difficulty: DifficultyLevel,
    seed: int = 42,
) -> Tuple[List[float], List[float]]:
    """Generate mixture with domain gap."""
    params = DIFFICULTY_PARAMS[difficulty]
    ppm_grid = np.arange(params.ppm_min, params.ppm_max, params.resolution)
    mixture = np.zeros_like(ppm_grid)

    np.random.seed(seed)

    for comp_name, weight in composition.items():
        if comp_name not in references:
            continue

        ref = references[comp_name]
        comp_seed = (seed + hash(comp_name)) % (2**31)

        # Generate PERTURBED spectrum
        perturbed = perturb_spectrum(
            ppm=ref["ppm"],
            intensity=ref["intensity"],
            difficulty=difficulty,
            seed=comp_seed,
            return_continuous=True,
        )

        mixture += weight * np.array(perturbed["intensity"])

    # Normalize and add noise
    if np.max(mixture) > 0:
        mixture = mixture / np.max(mixture)

    np.random.seed(seed + 9999)
    noise_std = 1.0 / params.snr
    mixture = mixture + np.random.normal(0, noise_std, size=mixture.shape)
    mixture = np.maximum(mixture, 0)

    if np.max(mixture) > 0:
        mixture = mixture / np.max(mixture)

    return ppm_grid.tolist(), mixture.tolist()


# ============================================================================
# Evaluation
# ============================================================================

@dataclass
class MethodResult:
    """Result for a single method on a single test case."""
    method: str
    test_id: str
    ground_truth: Dict[str, float]
    predicted: Dict[str, float]
    mae: float
    rmse: float
    max_error: float


def evaluate_method(
    method_name: str,
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    references: Dict[str, Dict],
    ground_truth: Dict[str, float],
    test_id: str,
    seed: int = 42,
) -> MethodResult:
    """Evaluate a single method on a single test case."""
    method_config = BASELINE_METHODS.get(method_name)
    if not method_config:
        raise ValueError(f"Unknown method: {method_name}")

    fn = method_config["fn"]

    # Run method
    if method_name == "random":
        predicted = fn(mixture_ppm, mixture_intensity, references, seed=seed)
    else:
        predicted = fn(mixture_ppm, mixture_intensity, references)

    # Calculate metrics
    errors = []
    for comp in ground_truth:
        true_val = ground_truth[comp]
        pred_val = predicted.get(comp, 0.0)
        errors.append(abs(true_val - pred_val))

    errors = np.array(errors)

    return MethodResult(
        method=method_name,
        test_id=test_id,
        ground_truth=ground_truth,
        predicted=predicted,
        mae=float(np.mean(errors)),
        rmse=float(np.sqrt(np.mean(errors**2))),
        max_error=float(np.max(errors)),
    )


def run_benchmark(
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
    methods: Optional[List[str]] = None,
    test_cases: Optional[List[Dict]] = None,
    seed: int = 42,
) -> Dict:
    """
    Run full benchmark comparing all methods.
    """
    if methods is None:
        methods = METHODS_TO_COMPARE
    if test_cases is None:
        test_cases = TEST_CASES

    print("=" * 70)
    print(f"BENCHMARK: MixSense vs Baselines ({difficulty.value.upper()})")
    print("=" * 70)
    print(f"\nMethods: {', '.join(methods)}")
    print(f"Test cases: {len(test_cases)}")
    print(f"Domain gap: {difficulty.value}")

    all_results = []

    # Run each test case
    for idx, test in enumerate(test_cases):
        print(f"\n--- {test['id']}: {test['name']} ---")

        # Generate mixture (use index for reproducible seed offset)
        mix_ppm, mix_int = generate_mixture_spectrum(
            composition=test["composition"],
            references=REFERENCE_SPECTRA,
            difficulty=difficulty,
            seed=seed + idx * 100,  # Deterministic offset
        )

        # Get relevant references
        test_refs = {
            name: REFERENCE_SPECTRA[name]
            for name in test["composition"]
            if name in REFERENCE_SPECTRA
        }

        # Evaluate each method
        method_results = {}
        for method in methods:
            result = evaluate_method(
                method_name=method,
                mixture_ppm=mix_ppm,
                mixture_intensity=mix_int,
                references=test_refs,
                ground_truth=test["composition"],
                test_id=test["id"],
                seed=seed,
            )
            method_results[method] = result
            all_results.append(result)

        # Print comparison for this test
        print(f"  Ground truth: {test['composition']}")
        print(f"  {'Method':<15} {'MAE':<10} {'Prediction'}")
        print(f"  {'-'*50}")
        for method in methods:
            r = method_results[method]
            pred_str = ", ".join(f"{k}:{v:.2f}" for k, v in r.predicted.items())
            print(f"  {method:<15} {r.mae:<10.4f} {pred_str}")

    # Summary by method
    print("\n" + "=" * 70)
    print("SUMMARY BY METHOD")
    print("=" * 70)

    method_summary = {}
    print(f"\n{'Method':<18} {'MAE Mean':<12} {'MAE Std':<12} {'RMSE Mean':<12}")
    print("-" * 54)

    for method in methods:
        method_results = [r for r in all_results if r.method == method]
        maes = [r.mae for r in method_results]
        rmses = [r.rmse for r in method_results]

        summary = {
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes)),
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses)),
        }
        method_summary[method] = summary

        print(f"{method:<18} {summary['mae_mean']:<12.4f} {summary['mae_std']:<12.4f} {summary['rmse_mean']:<12.4f}")

    # Rank methods
    print("\n" + "=" * 70)
    print("METHOD RANKING (by MAE)")
    print("=" * 70)

    ranked = sorted(method_summary.items(), key=lambda x: x[1]["mae_mean"])
    for i, (method, stats) in enumerate(ranked, 1):
        desc = BASELINE_METHODS[method]["description"]
        print(f"  {i}. {method:<15} MAE={stats['mae_mean']:.4f}  ({desc})")

    # Output
    output = {
        "timestamp": datetime.now().isoformat(),
        "difficulty": difficulty.value,
        "seed": seed,
        "n_tests": len(test_cases),
        "n_methods": len(methods),
        "method_summary": method_summary,
        "ranking": [m for m, _ in ranked],
        "results": [asdict(r) for r in all_results],
    }

    return output


def run_full_comparison(seed: int = 42) -> Dict:
    """Run benchmark at all difficulty levels."""
    print("\n" + "#" * 70)
    print("#  FULL COMPARISON: All Methods x All Difficulties")
    print("#" * 70)

    all_summaries = {}

    for difficulty in DifficultyLevel:
        result = run_benchmark(difficulty=difficulty, seed=seed)
        all_summaries[difficulty.value] = result["method_summary"]

    # Final comparison table
    print("\n" + "=" * 70)
    print("FINAL COMPARISON TABLE (MAE by Method x Difficulty)")
    print("=" * 70)

    methods = METHODS_TO_COMPARE
    print(f"\n{'Method':<18} {'Easy':<10} {'Medium':<10} {'Hard':<10}")
    print("-" * 48)

    for method in methods:
        row = f"{method:<18}"
        for diff in ["easy", "medium", "hard"]:
            if diff in all_summaries and method in all_summaries[diff]:
                mae = all_summaries[diff][method]["mae_mean"]
                row += f" {mae:<10.4f}"
            else:
                row += f" {'N/A':<10}"
        print(row)

    return {
        "timestamp": datetime.now().isoformat(),
        "by_difficulty": all_summaries,
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark MixSense vs baselines")
    parser.add_argument("--difficulty", "-d", choices=["easy", "medium", "hard", "all"],
                        default="medium", help="Difficulty level")
    parser.add_argument("--output", "-o", type=str, help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.difficulty == "all":
        results = run_full_comparison(seed=args.seed)
    else:
        difficulty = DifficultyLevel(args.difficulty)
        results = run_benchmark(difficulty=difficulty, seed=args.seed)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
