#!/usr/bin/env python3
"""
Baseline methods for NMR mixture deconvolution.

These baselines provide comparison points for evaluating the main
Magnetstein/optimal transport approach.

Baselines implemented:
1. NNLS (Non-Negative Least Squares) - Simple linear algebra
2. Peak Matching - Heuristic based on peak proximity
3. Integration Ratio - Classic NMR quantification by peak integration
4. Random Guess - Lower bound sanity check
5. Uniform Prior - Assume equal concentrations

Usage:
    from app.eval.baselines import (
        baseline_nnls,
        baseline_peak_matching,
        baseline_integration,
        baseline_random,
        baseline_uniform,
        run_all_baselines,
    )

    results = run_all_baselines(mixture_ppm, mixture_intensity, references)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.optimize import nnls, minimize
from dataclasses import dataclass


# ============================================================================
# Helper Functions
# ============================================================================

def lorentzian(x: np.ndarray, center: float, intensity: float, width: float = 0.02) -> np.ndarray:
    """Generate Lorentzian peak shape."""
    return intensity * (width**2) / ((x - center)**2 + width**2)


def generate_reference_spectrum(
    ppm_grid: np.ndarray,
    ref_ppm: List[float],
    ref_intensity: List[float],
    linewidth: float = 0.02,
) -> np.ndarray:
    """Generate continuous spectrum from peaks on given grid."""
    spectrum = np.zeros_like(ppm_grid)
    for ppm, intensity in zip(ref_ppm, ref_intensity):
        spectrum += lorentzian(ppm_grid, ppm, intensity, linewidth)
    # Normalize
    if np.max(spectrum) > 0:
        spectrum = spectrum / np.max(spectrum)
    return spectrum


def find_peaks(
    ppm: np.ndarray,
    intensity: np.ndarray,
    threshold: float = 0.05,
    min_distance: float = 0.05,
) -> List[Tuple[float, float]]:
    """
    Simple peak detection.

    Returns list of (ppm, intensity) for detected peaks.
    """
    peaks = []
    n = len(intensity)

    for i in range(1, n - 1):
        # Local maximum above threshold
        if intensity[i] > threshold and intensity[i] > intensity[i-1] and intensity[i] > intensity[i+1]:
            # Check minimum distance from existing peaks
            if all(abs(ppm[i] - p[0]) > min_distance for p in peaks):
                peaks.append((float(ppm[i]), float(intensity[i])))

    return sorted(peaks, key=lambda x: x[1], reverse=True)  # Sort by intensity


# ============================================================================
# Baseline 1: Non-Negative Least Squares (NNLS)
# ============================================================================

def baseline_nnls(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    references: Dict[str, Dict],
    linewidth: float = 0.02,
) -> Dict[str, float]:
    """
    Non-Negative Least Squares deconvolution.

    Solves: minimize ||A @ x - mixture||² subject to x >= 0
    where A is the matrix of reference spectra.

    This is a simple linear algebra approach that doesn't account for
    peak shifts or linewidth variations.

    Pros:
    - Fast, closed-form solution
    - Guaranteed non-negative results

    Cons:
    - Assumes perfect peak alignment
    - Sensitive to baseline and noise
    - No regularization
    """
    ppm_arr = np.array(mixture_ppm)
    mix_arr = np.array(mixture_intensity)

    ref_names = list(references.keys())
    n_refs = len(ref_names)
    n_points = len(ppm_arr)

    # Build design matrix
    A = np.zeros((n_points, n_refs))
    for j, name in enumerate(ref_names):
        ref = references[name]
        A[:, j] = generate_reference_spectrum(
            ppm_arr, ref["ppm"], ref["intensity"], linewidth
        )

    # Solve NNLS
    try:
        x, residual = nnls(A, mix_arr)
        # Normalize to sum to 1
        if np.sum(x) > 0:
            x = x / np.sum(x)
        return {name: float(x[i]) for i, name in enumerate(ref_names)}
    except Exception:
        return {name: 1.0 / n_refs for name in ref_names}


# ============================================================================
# Baseline 2: Peak Matching Heuristic
# ============================================================================

def baseline_peak_matching(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    references: Dict[str, Dict],
    match_tolerance: float = 0.1,
) -> Dict[str, float]:
    """
    Peak matching heuristic.

    For each reference, count how many of its peaks match peaks in the
    mixture, weighted by intensity. Normalize to get proportions.

    This mimics manual NMR interpretation where chemists look for
    characteristic peaks of each compound.

    Pros:
    - Intuitive, similar to manual analysis
    - Robust to baseline issues

    Cons:
    - Doesn't use full spectral information
    - Fails with overlapping peaks
    - No quantitative accuracy for similar compounds
    """
    ppm_arr = np.array(mixture_ppm)
    int_arr = np.array(mixture_intensity)

    # Find peaks in mixture
    mixture_peaks = find_peaks(ppm_arr, int_arr, threshold=0.05)

    if not mixture_peaks:
        # Fallback to uniform if no peaks detected
        n = len(references)
        return {name: 1.0 / n for name in references}

    scores = {}

    for name, ref in references.items():
        ref_ppm = ref["ppm"]
        ref_int = ref["intensity"]

        score = 0.0
        for ref_p, ref_i in zip(ref_ppm, ref_int):
            # Find closest mixture peak
            for mix_p, mix_i in mixture_peaks:
                if abs(ref_p - mix_p) < match_tolerance:
                    # Weight by both reference and mixture intensity
                    score += ref_i * mix_i
                    break

        scores[name] = score

    # Normalize
    total = sum(scores.values())
    if total > 0:
        return {name: score / total for name, score in scores.items()}
    else:
        n = len(references)
        return {name: 1.0 / n for name in references}


# ============================================================================
# Baseline 3: Integration Ratio
# ============================================================================

def baseline_integration(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    references: Dict[str, Dict],
    integration_width: float = 0.1,
) -> Dict[str, float]:
    """
    Integration-based quantification.

    For each reference's characteristic peak region, integrate the mixture
    spectrum in that region. Use the ratio of integrals as concentration.

    This is the classic NMR quantification approach using peak integrals.

    Pros:
    - Standard NMR quantification method
    - Well-understood by chemists

    Cons:
    - Requires non-overlapping peaks
    - Assumes known proton counts
    - Sensitive to baseline
    """
    ppm_arr = np.array(mixture_ppm)
    int_arr = np.array(mixture_intensity)

    integrals = {}

    for name, ref in references.items():
        ref_ppm = ref["ppm"]
        ref_int = ref["intensity"]
        protons = ref.get("protons", len(ref_ppm))

        # Find the most intense peak for this reference
        if not ref_ppm:
            integrals[name] = 0.0
            continue

        max_idx = np.argmax(ref_int)
        char_peak = ref_ppm[max_idx]

        # Integrate mixture around this peak
        mask = np.abs(ppm_arr - char_peak) < integration_width
        if np.any(mask):
            integral = np.trapz(int_arr[mask], ppm_arr[mask])
            # Normalize by proton count
            integrals[name] = integral / protons if protons > 0 else integral
        else:
            integrals[name] = 0.0

    # Normalize
    total = sum(integrals.values())
    if total > 0:
        return {name: val / total for name, val in integrals.items()}
    else:
        n = len(references)
        return {name: 1.0 / n for name in references}


# ============================================================================
# Baseline 4: Random Guess (Lower Bound)
# ============================================================================

def baseline_random(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    references: Dict[str, Dict],
    seed: Optional[int] = None,
) -> Dict[str, float]:
    """
    Random concentration assignment.

    This is a lower bound / sanity check. Any reasonable method should
    significantly outperform random guessing.

    Generates random proportions that sum to 1.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(references)
    # Dirichlet distribution gives random proportions summing to 1
    props = np.random.dirichlet(np.ones(n))

    return {name: float(props[i]) for i, name in enumerate(references)}


# ============================================================================
# Baseline 5: Uniform Prior
# ============================================================================

def baseline_uniform(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    references: Dict[str, Dict],
) -> Dict[str, float]:
    """
    Uniform concentration assumption.

    Assumes all components are present in equal amounts.
    This is a naive baseline that ignores the data entirely.

    Useful as a reference point - if a method doesn't beat uniform,
    it's not using the spectral information effectively.
    """
    n = len(references)
    return {name: 1.0 / n for name in references}


# ============================================================================
# Baseline 6: Regularized NNLS (L2)
# ============================================================================

def baseline_nnls_regularized(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    references: Dict[str, Dict],
    lambda_reg: float = 0.01,
    linewidth: float = 0.02,
) -> Dict[str, float]:
    """
    Regularized Non-Negative Least Squares.

    Solves: minimize ||A @ x - mixture||² + λ||x||²  subject to x >= 0

    L2 regularization (Ridge) helps with ill-conditioned problems and
    prevents overfitting to noise.

    Pros:
    - More stable than plain NNLS
    - Handles collinear references better

    Cons:
    - Requires tuning λ
    - Still assumes perfect alignment
    """
    ppm_arr = np.array(mixture_ppm)
    mix_arr = np.array(mixture_intensity)

    ref_names = list(references.keys())
    n_refs = len(ref_names)
    n_points = len(ppm_arr)

    # Build design matrix
    A = np.zeros((n_points, n_refs))
    for j, name in enumerate(ref_names):
        ref = references[name]
        A[:, j] = generate_reference_spectrum(
            ppm_arr, ref["ppm"], ref["intensity"], linewidth
        )

    # Augment for regularization: [A; sqrt(λ)I] @ x ≈ [mixture; 0]
    A_aug = np.vstack([A, np.sqrt(lambda_reg) * np.eye(n_refs)])
    b_aug = np.concatenate([mix_arr, np.zeros(n_refs)])

    try:
        x, _ = nnls(A_aug, b_aug)
        if np.sum(x) > 0:
            x = x / np.sum(x)
        return {name: float(x[i]) for i, name in enumerate(ref_names)}
    except Exception:
        return {name: 1.0 / n_refs for name in ref_names}


# ============================================================================
# Baseline 7: Sparse NNLS (L1)
# ============================================================================

def baseline_nnls_sparse(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    references: Dict[str, Dict],
    lambda_reg: float = 0.01,
    linewidth: float = 0.02,
) -> Dict[str, float]:
    """
    Sparse Non-Negative Least Squares (LASSO-style).

    Solves: minimize ||A @ x - mixture||² + λ||x||₁  subject to x >= 0

    L1 regularization encourages sparse solutions - useful when you
    expect only a few components to be present.

    Pros:
    - Automatic component selection
    - Good when mixture has few components

    Cons:
    - May miss minor components
    - Requires tuning λ
    """
    ppm_arr = np.array(mixture_ppm)
    mix_arr = np.array(mixture_intensity)

    ref_names = list(references.keys())
    n_refs = len(ref_names)

    # Build design matrix
    A = np.zeros((len(ppm_arr), n_refs))
    for j, name in enumerate(ref_names):
        ref = references[name]
        A[:, j] = generate_reference_spectrum(
            ppm_arr, ref["ppm"], ref["intensity"], linewidth
        )

    def objective(x):
        residual = np.sum((A @ x - mix_arr)**2)
        l1_penalty = lambda_reg * np.sum(x)
        return residual + l1_penalty

    # Bounded optimization (x >= 0)
    x0 = np.ones(n_refs) / n_refs
    bounds = [(0, None) for _ in range(n_refs)]

    try:
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        x = result.x
        if np.sum(x) > 0:
            x = x / np.sum(x)
        return {name: float(x[i]) for i, name in enumerate(ref_names)}
    except Exception:
        return {name: 1.0 / n_refs for name in ref_names}


# ============================================================================
# Run All Baselines
# ============================================================================

@dataclass
class BaselineResult:
    """Result from a single baseline method."""
    name: str
    description: str
    predictions: Dict[str, float]


BASELINE_METHODS = {
    "nnls": {
        "fn": baseline_nnls,
        "description": "Non-Negative Least Squares - simple linear algebra",
    },
    "nnls_l2": {
        "fn": baseline_nnls_regularized,
        "description": "NNLS with L2 regularization (Ridge)",
    },
    "nnls_l1": {
        "fn": baseline_nnls_sparse,
        "description": "NNLS with L1 regularization (sparse/LASSO)",
    },
    "peak_matching": {
        "fn": baseline_peak_matching,
        "description": "Heuristic peak matching by proximity",
    },
    "integration": {
        "fn": baseline_integration,
        "description": "Classic NMR integration ratio",
    },
    "uniform": {
        "fn": baseline_uniform,
        "description": "Uniform prior (equal concentrations)",
    },
    "random": {
        "fn": baseline_random,
        "description": "Random guess (lower bound)",
    },
}


def run_all_baselines(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    references: Dict[str, Dict],
    seed: int = 42,
) -> Dict[str, BaselineResult]:
    """
    Run all baseline methods on a mixture.

    Returns dict of {method_name: BaselineResult}
    """
    results = {}

    for name, config in BASELINE_METHODS.items():
        fn = config["fn"]
        desc = config["description"]

        if name == "random":
            preds = fn(mixture_ppm, mixture_intensity, references, seed=seed)
        else:
            preds = fn(mixture_ppm, mixture_intensity, references)

        results[name] = BaselineResult(
            name=name,
            description=desc,
            predictions=preds,
        )

    return results


# ============================================================================
# Demo / Testing
# ============================================================================

if __name__ == "__main__":
    print("Baseline Methods Demo")
    print("=" * 60)

    # Simple test case
    import numpy as np

    # Generate synthetic mixture (toluene + benzene, 60:40)
    ppm_grid = np.arange(0, 12, 0.01)

    refs = {
        "toluene": {"ppm": [7.20, 7.15, 2.35], "intensity": [0.6, 0.4, 1.0], "protons": 8},
        "benzene": {"ppm": [7.36], "intensity": [1.0], "protons": 6},
    }

    # Generate mixture
    mixture = np.zeros_like(ppm_grid)
    mixture += 0.6 * generate_reference_spectrum(ppm_grid, refs["toluene"]["ppm"], refs["toluene"]["intensity"])
    mixture += 0.4 * generate_reference_spectrum(ppm_grid, refs["benzene"]["ppm"], refs["benzene"]["intensity"])
    mixture = mixture / np.max(mixture)
    mixture += np.random.normal(0, 0.01, size=mixture.shape)
    mixture = np.maximum(mixture, 0)

    ground_truth = {"toluene": 0.6, "benzene": 0.4}

    print(f"\nGround truth: {ground_truth}")
    print("\nBaseline results:")
    print("-" * 60)

    results = run_all_baselines(ppm_grid.tolist(), mixture.tolist(), refs)

    for name, result in results.items():
        preds = result.predictions
        mae = np.mean([abs(preds[k] - ground_truth[k]) for k in ground_truth])
        print(f"\n{name}:")
        print(f"  {result.description}")
        print(f"  Predictions: {{{', '.join(f'{k}: {v:.3f}' for k, v in preds.items())}}}")
        print(f"  MAE: {mae:.4f}")
