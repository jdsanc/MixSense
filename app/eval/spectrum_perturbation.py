#!/usr/bin/env python3
"""
Realistic NMR spectrum perturbation for publication-quality evaluation.

This module introduces controlled domain gap between "experimental" spectra
(used to generate mixtures) and reference spectra (used for deconvolution).

Perturbation sources modeled:
1. Peak shift jitter (solvent effects, temperature, concentration)
2. Linewidth variation (shimming, relaxation, sample purity)
3. Intensity variation (integration errors, relaxation effects)
4. Baseline drift (probe background, impurities)
5. Random noise (instrument noise, digitization)

Usage:
    from app.eval.spectrum_perturbation import perturb_spectrum, DifficultyLevel

    # Get perturbed "experimental" spectrum
    exp_spectrum = perturb_spectrum(
        ppm=[7.26, 6.90, 3.78],
        intensity=[2.0, 3.0, 3.0],
        difficulty=DifficultyLevel.MEDIUM,
        seed=42
    )
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class DifficultyLevel(Enum):
    """Difficulty levels for synthetic evaluation."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class PerturbationParams:
    """Parameters controlling spectrum perturbation."""
    # Peak shift (ppm)
    shift_mean: float = 0.0        # Systematic shift
    shift_std: float = 0.02        # Random jitter std

    # Linewidth (multiplicative factor)
    linewidth_mean: float = 1.0    # Mean linewidth factor
    linewidth_std: float = 0.1     # Linewidth variation

    # Intensity (multiplicative factor)
    intensity_mean: float = 1.0    # Mean intensity factor
    intensity_std: float = 0.05    # Intensity variation

    # Baseline drift
    baseline_amplitude: float = 0.01  # Max baseline drift (fraction of max)
    baseline_frequency: float = 0.5   # Baseline curvature

    # Noise
    snr: float = 200.0             # Signal-to-noise ratio

    # Spectral resolution
    resolution: float = 0.001      # ppm per point
    ppm_min: float = 0.0
    ppm_max: float = 12.0
    default_linewidth: float = 0.02  # Default peak FWHM


# Pre-defined difficulty levels
DIFFICULTY_PARAMS = {
    DifficultyLevel.EASY: PerturbationParams(
        shift_std=0.01,
        linewidth_std=0.05,
        intensity_std=0.03,
        baseline_amplitude=0.005,
        snr=500.0,
    ),
    DifficultyLevel.MEDIUM: PerturbationParams(
        shift_std=0.03,
        linewidth_std=0.15,
        intensity_std=0.10,
        baseline_amplitude=0.02,
        snr=100.0,
    ),
    DifficultyLevel.HARD: PerturbationParams(
        shift_std=0.05,
        linewidth_std=0.30,
        intensity_std=0.20,
        baseline_amplitude=0.05,
        snr=50.0,
    ),
}


def lorentzian(x: np.ndarray, center: float, intensity: float, width: float) -> np.ndarray:
    """Generate Lorentzian peak shape (typical NMR lineshape)."""
    return intensity * (width**2) / ((x - center)**2 + width**2)


def gaussian(x: np.ndarray, center: float, intensity: float, width: float) -> np.ndarray:
    """Generate Gaussian peak shape."""
    return intensity * np.exp(-0.5 * ((x - center) / width)**2)


def voigt(x: np.ndarray, center: float, intensity: float,
          width_l: float, width_g: float) -> np.ndarray:
    """
    Pseudo-Voigt profile (more realistic NMR lineshape).
    Approximation: linear combination of Lorentzian and Gaussian.
    """
    eta = 0.5  # Mixing parameter (0=pure Gaussian, 1=pure Lorentzian)
    return eta * lorentzian(x, center, intensity, width_l) + \
           (1 - eta) * gaussian(x, center, intensity, width_g)


def generate_baseline(
    ppm_grid: np.ndarray,
    amplitude: float = 0.01,
    frequency: float = 0.5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate realistic baseline drift.

    Uses low-frequency polynomial + sinusoidal components.
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(ppm_grid)
    x_norm = (ppm_grid - ppm_grid.min()) / (ppm_grid.max() - ppm_grid.min())

    # Polynomial component (slow drift)
    poly_coeffs = np.random.normal(0, amplitude / 3, size=3)
    baseline = (poly_coeffs[0] +
                poly_coeffs[1] * x_norm +
                poly_coeffs[2] * x_norm**2)

    # Sinusoidal component (rolling baseline)
    phase = np.random.uniform(0, 2 * np.pi)
    baseline += amplitude * 0.5 * np.sin(2 * np.pi * frequency * x_norm + phase)

    return baseline


def perturb_peaks(
    ppm: List[float],
    intensity: List[float],
    params: PerturbationParams,
    seed: Optional[int] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Apply perturbations to peak positions, intensities, and linewidths.

    Returns:
        (perturbed_ppm, perturbed_intensity, linewidths)
    """
    if seed is not None:
        np.random.seed(seed)

    n_peaks = len(ppm)

    # Shift perturbation
    shifts = np.random.normal(params.shift_mean, params.shift_std, n_peaks)
    perturbed_ppm = [p + s for p, s in zip(ppm, shifts)]

    # Intensity perturbation (multiplicative)
    int_factors = np.random.lognormal(
        np.log(params.intensity_mean),
        params.intensity_std,
        n_peaks
    )
    perturbed_intensity = [i * f for i, f in zip(intensity, int_factors)]

    # Linewidth perturbation
    lw_factors = np.random.lognormal(
        np.log(params.linewidth_mean),
        params.linewidth_std,
        n_peaks
    )
    linewidths = [params.default_linewidth * f for f in lw_factors]

    return perturbed_ppm, perturbed_intensity, linewidths


def perturb_spectrum(
    ppm: List[float],
    intensity: List[float],
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
    params: Optional[PerturbationParams] = None,
    seed: Optional[int] = None,
    return_continuous: bool = True,
) -> Dict:
    """
    Apply realistic perturbations to an NMR spectrum.

    Args:
        ppm: Peak positions (chemical shifts)
        intensity: Peak intensities/integrals
        difficulty: Easy/Medium/Hard perturbation level
        params: Custom perturbation parameters (overrides difficulty)
        seed: Random seed for reproducibility
        return_continuous: If True, return dense spectrum; else return peaks

    Returns:
        Dict with:
            - ppm: Peak positions or grid
            - intensity: Peak intensities or continuous spectrum
            - linewidths: Peak linewidths (if return_continuous=False)
            - params: Applied parameters
            - seed: Random seed used
    """
    if params is None:
        params = DIFFICULTY_PARAMS[difficulty]

    # Set master seed
    if seed is not None:
        np.random.seed(seed)
        peak_seed = seed
        baseline_seed = seed + 1
        noise_seed = seed + 2
    else:
        peak_seed = None
        baseline_seed = None
        noise_seed = None

    # Perturb peaks
    pert_ppm, pert_int, linewidths = perturb_peaks(
        ppm, intensity, params, seed=peak_seed
    )

    if not return_continuous:
        return {
            "ppm": pert_ppm,
            "intensity": pert_int,
            "linewidths": linewidths,
            "params": params,
            "seed": seed,
        }

    # Generate continuous spectrum
    ppm_grid = np.arange(params.ppm_min, params.ppm_max, params.resolution)
    spectrum = np.zeros_like(ppm_grid)

    for p, i, w in zip(pert_ppm, pert_int, linewidths):
        if params.ppm_min <= p <= params.ppm_max:
            spectrum += lorentzian(ppm_grid, p, i, w)

    # Normalize to max = 1 before adding artifacts
    max_signal = np.max(spectrum)
    if max_signal > 0:
        spectrum = spectrum / max_signal

    # Add baseline drift
    baseline = generate_baseline(
        ppm_grid,
        amplitude=params.baseline_amplitude,
        frequency=params.baseline_frequency,
        seed=baseline_seed,
    )
    spectrum = spectrum + baseline

    # Add noise
    if noise_seed is not None:
        np.random.seed(noise_seed)
    noise_std = 1.0 / params.snr
    noise = np.random.normal(0, noise_std, size=spectrum.shape)
    spectrum = spectrum + noise

    # Clip negative values
    spectrum = np.maximum(spectrum, 0)

    # Re-normalize
    if np.max(spectrum) > 0:
        spectrum = spectrum / np.max(spectrum)

    return {
        "ppm": ppm_grid.tolist(),
        "intensity": spectrum.tolist(),
        "peak_ppm": pert_ppm,
        "peak_intensity": pert_int,
        "linewidths": linewidths,
        "params": params,
        "seed": seed,
    }


def generate_mixture_with_gap(
    components: Dict[str, float],
    reference_spectra: Dict[str, Dict],
    difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
    seed: Optional[int] = None,
) -> Tuple[Dict, Dict[str, Dict]]:
    """
    Generate a mixture spectrum with domain gap from references.

    The mixture is generated from PERTURBED spectra, while the
    references remain unperturbed. This simulates the difference
    between experimental data and database references.

    Args:
        components: {compound_name: weight}
        reference_spectra: {compound_name: {ppm: [...], intensity: [...]}}
        difficulty: Perturbation level
        seed: Random seed

    Returns:
        (mixture_spectrum, original_references)
    """
    params = DIFFICULTY_PARAMS[difficulty]
    ppm_grid = np.arange(params.ppm_min, params.ppm_max, params.resolution)
    mixture = np.zeros_like(ppm_grid)

    if seed is not None:
        np.random.seed(seed)

    for name, weight in components.items():
        if name not in reference_spectra:
            print(f"Warning: {name} not in references, skipping")
            continue

        ref = reference_spectra[name]
        component_seed = hash(name) % (2**31) if seed is not None else None

        # Generate PERTURBED spectrum for this component
        perturbed = perturb_spectrum(
            ppm=ref["ppm"],
            intensity=ref["intensity"],
            difficulty=difficulty,
            seed=component_seed,
            return_continuous=True,
        )

        # Add weighted contribution to mixture
        mixture += weight * np.array(perturbed["intensity"])

    # Normalize
    if np.max(mixture) > 0:
        mixture = mixture / np.max(mixture)

    # Add mixture-level noise (instrument noise)
    if seed is not None:
        np.random.seed(seed + 999)
    noise_std = 1.0 / params.snr
    mixture = mixture + np.random.normal(0, noise_std, size=mixture.shape)
    mixture = np.maximum(mixture, 0)

    if np.max(mixture) > 0:
        mixture = mixture / np.max(mixture)

    mixture_spectrum = {
        "ppm": ppm_grid.tolist(),
        "intensity": mixture.tolist(),
        "ground_truth": components,
        "difficulty": difficulty.value,
        "seed": seed,
    }

    return mixture_spectrum, reference_spectra


# ============================================================================
# Convenience functions
# ============================================================================

def compare_spectra(
    original: Dict,
    perturbed: Dict,
    metric: str = "rmse",
) -> float:
    """
    Compare two spectra to quantify domain gap.

    Args:
        original: Original spectrum {ppm, intensity}
        perturbed: Perturbed spectrum {ppm, intensity}
        metric: "rmse", "mae", or "cosine"

    Returns:
        Similarity/distance metric
    """
    # Ensure same grid
    orig_int = np.array(original["intensity"])
    pert_int = np.array(perturbed["intensity"])

    if len(orig_int) != len(pert_int):
        raise ValueError("Spectra must have same length")

    if metric == "rmse":
        return float(np.sqrt(np.mean((orig_int - pert_int)**2)))
    elif metric == "mae":
        return float(np.mean(np.abs(orig_int - pert_int)))
    elif metric == "cosine":
        dot = np.dot(orig_int, pert_int)
        norm = np.linalg.norm(orig_int) * np.linalg.norm(pert_int)
        return float(dot / norm) if norm > 0 else 0.0
    else:
        raise ValueError(f"Unknown metric: {metric}")


if __name__ == "__main__":
    # Demo
    print("Spectrum Perturbation Demo")
    print("=" * 50)

    # Example peaks (anisole-like)
    ppm = [7.28, 6.92, 3.80]
    intensity = [2.0, 3.0, 3.0]

    for difficulty in DifficultyLevel:
        result = perturb_spectrum(
            ppm, intensity,
            difficulty=difficulty,
            seed=42,
            return_continuous=False
        )
        print(f"\n{difficulty.value.upper()}:")
        print(f"  Original ppm:   {ppm}")
        print(f"  Perturbed ppm:  {[f'{p:.3f}' for p in result['ppm']]}")
        print(f"  Linewidths:     {[f'{w:.4f}' for w in result['linewidths']]}")
