#!/usr/bin/env python3
"""
Publication-quality evaluation with realistic domain gap simulation.

Key insight: The "experimental" mixture is generated from PERTURBED spectra,
while deconvolution uses UNPERTURBED references. This simulates the real-world
difference between experimental data and database references.

Usage:
    python -m app.eval.realistic_evaluation
    python -m app.eval.realistic_evaluation --difficulty medium
    python -m app.eval.realistic_evaluation --output results.json
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
    generate_mixture_with_gap,
    lorentzian,
)
from app.eval.predict_nmr import get_reference_with_fallback, predict_nmr_spectrum


# ============================================================================
# Test Case Definitions
# ============================================================================

# Reference spectra for test compounds (from NMRBank or literature)
# These serve as the "database" references for deconvolution
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

# Test cases with ground truth
DECONVOLUTION_TESTS = [
    {
        "id": "D1",
        "name": "Binary equal (50:50)",
        "composition": {"toluene": 0.5, "benzene": 0.5},
        "category": "binary",
    },
    {
        "id": "D2",
        "name": "Binary unequal (70:30)",
        "composition": {"toluene": 0.7, "benzene": 0.3},
        "category": "binary",
    },
    {
        "id": "D3",
        "name": "Trace detection (90:10)",
        "composition": {"toluene": 0.9, "benzene": 0.1},
        "category": "trace",
    },
    {
        "id": "D4",
        "name": "Trace detection (95:5)",
        "composition": {"toluene": 0.95, "benzene": 0.05},
        "category": "trace",
    },
    {
        "id": "D5",
        "name": "Ternary equal",
        "composition": {"toluene": 0.34, "chlorobenzene": 0.33, "benzene": 0.33},
        "category": "ternary",
    },
    {
        "id": "D6",
        "name": "Ternary gradient",
        "composition": {"toluene": 0.5, "chlorobenzene": 0.3, "benzene": 0.2},
        "category": "ternary",
    },
    {
        "id": "D7",
        "name": "Bromination partial (60:40)",
        "composition": {"anisole": 0.6, "p_bromoanisole": 0.4},
        "category": "reaction",
    },
    {
        "id": "D8",
        "name": "Bromination near-complete (10:90)",
        "composition": {"anisole": 0.1, "p_bromoanisole": 0.9},
        "category": "reaction",
    },
    {
        "id": "D9",
        "name": "Esterification mixture",
        "composition": {"acetic_acid": 0.3, "ethanol": 0.2, "ethyl_acetate": 0.5},
        "category": "reaction",
    },
    {
        "id": "D10",
        "name": "Four components",
        "composition": {"toluene": 0.4, "benzene": 0.25, "chlorobenzene": 0.2, "bromobenzene": 0.15},
        "category": "complex",
    },
]


# ============================================================================
# Spectrum Generation with Domain Gap
# ============================================================================

def generate_continuous_spectrum(
    ppm_peaks: List[float],
    intensity_peaks: List[float],
    ppm_min: float = 0.0,
    ppm_max: float = 12.0,
    resolution: float = 0.001,
    linewidth: float = 0.02,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate continuous spectrum from peaks."""
    ppm_grid = np.arange(ppm_min, ppm_max, resolution)
    spectrum = np.zeros_like(ppm_grid)

    for ppm, intensity in zip(ppm_peaks, intensity_peaks):
        if ppm_min <= ppm <= ppm_max:
            spectrum += lorentzian(ppm_grid, ppm, intensity, linewidth)

    return ppm_grid, spectrum


def generate_mixture_spectrum(
    composition: Dict[str, float],
    references: Dict[str, Dict],
    difficulty: DifficultyLevel,
    seed: int = 42,
) -> Tuple[List[float], List[float], Dict]:
    """
    Generate mixture spectrum with domain gap.

    The mixture is generated from PERTURBED versions of the references,
    simulating experimental variation.
    """
    params = DIFFICULTY_PARAMS[difficulty]
    ppm_grid = np.arange(params.ppm_min, params.ppm_max, params.resolution)
    mixture = np.zeros_like(ppm_grid)

    np.random.seed(seed)
    component_info = {}

    for comp_name, weight in composition.items():
        if comp_name not in references:
            print(f"Warning: {comp_name} not in references")
            continue

        ref = references[comp_name]

        # Generate PERTURBED spectrum for this component
        # (This is what makes it a realistic evaluation)
        comp_seed = (seed + hash(comp_name)) % (2**31)
        perturbed = perturb_spectrum(
            ppm=ref["ppm"],
            intensity=ref["intensity"],
            difficulty=difficulty,
            seed=comp_seed,
            return_continuous=True,
        )

        # Add weighted contribution
        pert_intensity = np.array(perturbed["intensity"])
        mixture += weight * pert_intensity

        component_info[comp_name] = {
            "weight": weight,
            "original_peaks": ref["ppm"],
            "perturbed_peaks": perturbed["peak_ppm"],
        }

    # Normalize
    if np.max(mixture) > 0:
        mixture = mixture / np.max(mixture)

    # Add mixture-level noise
    np.random.seed(seed + 9999)
    noise_std = 1.0 / params.snr
    mixture = mixture + np.random.normal(0, noise_std, size=mixture.shape)
    mixture = np.maximum(mixture, 0)

    if np.max(mixture) > 0:
        mixture = mixture / np.max(mixture)

    metadata = {
        "composition": composition,
        "difficulty": difficulty.value,
        "seed": seed,
        "components": component_info,
    }

    return ppm_grid.tolist(), mixture.tolist(), metadata


# ============================================================================
# Deconvolution (Simplified for Testing)
# ============================================================================

def simple_deconvolve(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    references: Dict[str, Dict],
    method: str = "nnls",
) -> Dict[str, float]:
    """
    Simple deconvolution using non-negative least squares.

    For real evaluation, use tools_magnetstein.quantify_single()
    """
    from scipy.optimize import nnls

    ppm_arr = np.array(mixture_ppm)
    mix_arr = np.array(mixture_intensity)

    # Build reference matrix
    # Each column is a reference spectrum interpolated to mixture grid
    ref_names = list(references.keys())
    n_refs = len(ref_names)
    n_points = len(ppm_arr)

    A = np.zeros((n_points, n_refs))

    for j, name in enumerate(ref_names):
        ref = references[name]
        # Generate continuous spectrum at mixture resolution
        ref_spectrum = np.zeros(n_points)
        for ppm_val, int_val in zip(ref["ppm"], ref["intensity"]):
            ref_spectrum += lorentzian(ppm_arr, ppm_val, int_val, width=0.02)

        # Normalize
        if np.max(ref_spectrum) > 0:
            ref_spectrum = ref_spectrum / np.max(ref_spectrum)

        A[:, j] = ref_spectrum

    # Solve: minimize ||A @ x - mixture||^2 s.t. x >= 0
    try:
        x, residual = nnls(A, mix_arr)

        # Normalize to sum to 1
        if np.sum(x) > 0:
            x = x / np.sum(x)

        return {name: float(x[i]) for i, name in enumerate(ref_names)}
    except Exception as e:
        print(f"Deconvolution failed: {e}")
        return {name: 0.0 for name in ref_names}


# ============================================================================
# Evaluation Metrics
# ============================================================================

@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    test_id: str
    test_name: str
    category: str
    difficulty: str
    ground_truth: Dict[str, float]
    predicted: Dict[str, float]
    mae: float
    rmse: float
    max_error: float
    per_component: Dict[str, float] = field(default_factory=dict)


def calculate_metrics(
    ground_truth: Dict[str, float],
    predicted: Dict[str, float],
) -> Tuple[float, float, float, Dict[str, float]]:
    """
    Calculate evaluation metrics.

    Returns (MAE, RMSE, max_error, per_component_errors)
    """
    all_components = set(ground_truth.keys()) | set(predicted.keys())

    errors = []
    per_component = {}

    for comp in all_components:
        true_val = ground_truth.get(comp, 0.0)
        pred_val = predicted.get(comp, 0.0)
        error = abs(true_val - pred_val)
        errors.append(error)
        per_component[comp] = error

    errors = np.array(errors)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_error = float(np.max(errors))

    return mae, rmse, max_error, per_component


# ============================================================================
# Main Evaluation
# ============================================================================

def run_evaluation(
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
    tests: Optional[List[Dict]] = None,
    seed: int = 42,
    use_magnetstein: bool = False,
) -> Dict:
    """
    Run full evaluation suite.

    Args:
        difficulty: Easy/Medium/Hard perturbation level
        tests: List of test cases (uses DECONVOLUTION_TESTS if None)
        seed: Random seed
        use_magnetstein: If True, try to use real Magnetstein

    Returns:
        Dict with all results and summary statistics
    """
    print("=" * 70)
    print(f"MixSense Realistic Evaluation - {difficulty.value.upper()} difficulty")
    print("=" * 70)
    print(f"\nKey: Mixture generated with perturbations, deconvolution uses clean refs")
    print(f"     This simulates experimental vs database spectrum differences\n")

    if tests is None:
        tests = DECONVOLUTION_TESTS

    # Try to use real deconvolution
    deconvolve_fn = simple_deconvolve
    if use_magnetstein:
        try:
            from app.tools_magnetstein import quantify_single, _HAS_MAGNETSTEIN
            if _HAS_MAGNETSTEIN:
                print("Using Magnetstein for deconvolution")
                # Wrap quantify_single to match our interface
                def magnetstein_wrapper(mix_ppm, mix_int, refs):
                    library = [
                        {"name": name, "ppm": r["ppm"], "intensity": r["intensity"]}
                        for name, r in refs.items()
                    ]
                    result = quantify_single(mix_ppm, mix_int, library, min_peaks=1)
                    return result.get("concentrations", {})
                deconvolve_fn = magnetstein_wrapper
        except ImportError:
            pass

    results = []
    all_mae = []
    all_rmse = []

    for idx, test in enumerate(tests):
        print(f"\n[{test['id']}] {test['name']}")
        print(f"    Ground truth: {test['composition']}")

        # Generate mixture with domain gap (use index for reproducible seed)
        mix_ppm, mix_int, meta = generate_mixture_spectrum(
            composition=test["composition"],
            references=REFERENCE_SPECTRA,
            difficulty=difficulty,
            seed=seed + idx * 100,  # Deterministic offset
        )

        # Get only the references we need
        test_refs = {
            name: REFERENCE_SPECTRA[name]
            for name in test["composition"].keys()
            if name in REFERENCE_SPECTRA
        }

        # Run deconvolution with CLEAN references
        predicted = deconvolve_fn(mix_ppm, mix_int, test_refs)

        # Calculate metrics
        mae, rmse, max_err, per_comp = calculate_metrics(
            test["composition"],
            predicted
        )

        print(f"    Predicted:    {{{', '.join(f'{k}: {v:.3f}' for k, v in predicted.items())}}}")
        print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, Max: {max_err:.4f}")

        all_mae.append(mae)
        all_rmse.append(rmse)

        results.append(EvaluationResult(
            test_id=test["id"],
            test_name=test["name"],
            category=test["category"],
            difficulty=difficulty.value,
            ground_truth=test["composition"],
            predicted=predicted,
            mae=mae,
            rmse=rmse,
            max_error=max_err,
            per_component=per_comp,
        ))

    # Summary by category
    print("\n" + "=" * 70)
    print("SUMMARY BY CATEGORY")
    print("=" * 70)

    categories = set(t["category"] for t in tests)
    category_summary = {}

    for cat in sorted(categories):
        cat_results = [r for r in results if r.category == cat]
        cat_mae = [r.mae for r in cat_results]
        print(f"\n{cat.upper()}:")
        print(f"  n = {len(cat_results)}")
        print(f"  MAE: {np.mean(cat_mae):.4f} ± {np.std(cat_mae):.4f}")
        category_summary[cat] = {
            "n": len(cat_results),
            "mae_mean": float(np.mean(cat_mae)),
            "mae_std": float(np.std(cat_mae)),
        }

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"\nDifficulty: {difficulty.value}")
    print(f"Total tests: {len(results)}")
    print(f"Mean MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    print(f"Mean RMSE: {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")

    output = {
        "timestamp": datetime.now().isoformat(),
        "difficulty": difficulty.value,
        "seed": seed,
        "n_tests": len(results),
        "summary": {
            "mae_mean": float(np.mean(all_mae)),
            "mae_std": float(np.std(all_mae)),
            "rmse_mean": float(np.mean(all_rmse)),
            "rmse_std": float(np.std(all_rmse)),
        },
        "by_category": category_summary,
        "results": [asdict(r) for r in results],
    }

    return output


def run_difficulty_comparison(seed: int = 42) -> Dict:
    """
    Run evaluation at all difficulty levels for comparison.
    """
    print("\n" + "#" * 70)
    print("#  DIFFICULTY LEVEL COMPARISON")
    print("#" * 70)

    all_results = {}

    for difficulty in DifficultyLevel:
        print(f"\n{'='*70}")
        print(f"  {difficulty.value.upper()} DIFFICULTY")
        print("=" * 70)

        result = run_evaluation(difficulty=difficulty, seed=seed)
        all_results[difficulty.value] = result["summary"]

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{'Difficulty':<12} {'MAE':<12} {'RMSE':<12}")
    print("-" * 36)
    for diff in ["easy", "medium", "hard"]:
        if diff in all_results:
            s = all_results[diff]
            print(f"{diff:<12} {s['mae_mean']:.4f}±{s['mae_std']:.4f}  {s['rmse_mean']:.4f}±{s['rmse_std']:.4f}")

    return all_results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run publication-quality evaluation with domain gap simulation"
    )
    parser.add_argument(
        "--difficulty", "-d",
        choices=["easy", "medium", "hard", "all"],
        default="medium",
        help="Difficulty level"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--magnetstein",
        action="store_true",
        help="Try to use Magnetstein for deconvolution"
    )

    args = parser.parse_args()

    if args.difficulty == "all":
        results = run_difficulty_comparison(seed=args.seed)
    else:
        difficulty = DifficultyLevel(args.difficulty)
        results = run_evaluation(
            difficulty=difficulty,
            seed=args.seed,
            use_magnetstein=args.magnetstein
        )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
