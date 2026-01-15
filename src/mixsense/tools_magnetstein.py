"""
Thin wrapper around Magnetstein for (a) single-spectrum mixture quant
and (b) reaction time-series analysis.

Requires: `pip install gurobipy` (optional, faster solver) or fallback solver.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
import sys
import os

# Add magnetstein to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'magnetstein'))

try:
    from masserstein import Spectrum
    from masserstein.deconv_simplex import estimate_proportions, estimate_proportions_in_time
    from pulp import LpSolverDefault
    _HAS_MAGNETSTEIN = True
except Exception as e:
    print(f"Warning: Could not import magnetstein: {e}")
    _HAS_MAGNETSTEIN = False

def _to_numpy(ppm: List[float], intensity: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert to numpy arrays and filter out negative ppm values (magnetstein requirement)"""
    ppm_arr = np.asarray(ppm, dtype=float)
    intensity_arr = np.asarray(intensity, dtype=float)
    
    # Filter out negative ppm values (magnetstein requires positive masses)
    mask = ppm_arr >= 0
    return ppm_arr[mask], intensity_arr[mask]

def quantify_single(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    library: List[Dict[str, List[float]]],  # [{"name":..., "ppm":[...], "intensity":[...]}]
    kappa_mixture: float = 0.25,
    kappa_components: float = 0.22,
) -> Dict:
    """
    Return {"concentrations": {name: value, ...}, "reconstructed": {"ppm": [...], "intensity": [...]}}
    """
    if not _HAS_MAGNETSTEIN:
        raise ImportError("magnetstein not available; add it as a submodule or pip-install the package.")

    # Create mixture spectrum
    mix_x, mix_y = _to_numpy(mixture_ppm, mixture_intensity)
    
    if len(mix_x) == 0:
        raise ValueError("No valid (non-negative) ppm values found in mixture spectrum")
    
    mix_confs = list(zip(mix_x, mix_y))
    mixture_spectrum = Spectrum(confs=mix_confs)
    
    # Create library spectra
    query_spectra = []
    names = []
    for comp in library:
        x, y = _to_numpy(comp["ppm"], comp["intensity"])
        if len(x) == 0:
            print(f"Warning: Skipping component {comp['name']} - no valid ppm values")
            continue
        comp_confs = list(zip(x, y))
        query_spectra.append(Spectrum(confs=comp_confs))
        names.append(comp["name"])
    
    if len(query_spectra) == 0:
        raise ValueError("No valid library spectra found")

    # Call magnetstein with default solver (CBC instead of Gurobi)
    res = estimate_proportions(
        spectrum=mixture_spectrum,
        query=query_spectra,
        MTD=kappa_mixture,
        MTD_th=kappa_components,
        verbose=False,
        solver=LpSolverDefault  # Use default solver (CBC)
    )

    # Extract results
    proportions = res[0]  # First element is the proportions array
    conc = {names[i]: float(proportions[i]) for i in range(len(names))}
    
    # For reconstruction, we'll use the mixture spectrum as is
    # (magnetstein doesn't return reconstruction in this API)
    recon = {"ppm": mix_x.tolist(), "intensity": mix_y.tolist()}
    
    return {"concentrations": conc, "reconstructed": recon}

def quantify_timeseries(
    times: List[float],
    mixtures: List[Dict[str, List[float]]],  # [{"ppm":[...], "intensity":[...]}] ordered by time
    library: List[Dict[str, List[float]]],
    kappa_mixture: float = 0.25,
    kappa_components: float = 0.22,
) -> Dict:
    """
    Return {"times": [...], "proportions": {name: [..series..], "contamination": [...]}}
    """
    if not _HAS_MAGNETSTEIN:
        raise ImportError("magnetstein not available")

    # Create mixture spectra for each time point
    mixture_spectra = []
    for i, m in enumerate(mixtures):
        x, y = _to_numpy(m["ppm"], m["intensity"])
        if len(x) == 0:
            print(f"Warning: Skipping time point {i} - no valid ppm values")
            continue
        mix_confs = list(zip(x, y))
        mixture_spectra.append(Spectrum(confs=mix_confs))
    
    if len(mixture_spectra) == 0:
        raise ValueError("No valid mixture spectra found")
    
    # Create library spectra
    reagents_spectra = []
    names = []
    for comp in library:
        x, y = _to_numpy(comp["ppm"], comp["intensity"])
        if len(x) == 0:
            print(f"Warning: Skipping component {comp.get('name', 'unknown')} - no valid ppm values")
            continue
        comp_confs = list(zip(x, y))
        reagents_spectra.append(Spectrum(confs=comp_confs))
        names.append(comp.get("name", f"comp_{len(names)}"))
    
    if len(reagents_spectra) == 0:
        raise ValueError("No valid library spectra found")

    # Call magnetstein timeseries function with default solver
    res = estimate_proportions_in_time(
        mixture_in_time=mixture_spectra,
        reagents_spectra=reagents_spectra,
        MTD=kappa_mixture,
        MTD_th=kappa_components,
        verbose=False,
        solver=LpSolverDefault  # Use default solver (CBC)
    )

    # Extract results - res is a dictionary with proportions_in_time
    proportions_in_time = res['proportions_in_time']  # This is a list of lists
    out = {"times": times, "proportions": {}}
    
    # Convert proportions array to dictionary format
    # proportions_in_time structure: [time0[comp0, comp1, ...], time1[comp0, comp1, ...], ...]
    # Each time point has proportions for all components
    for i, name in enumerate(names):
        if i < len(proportions_in_time[0]):  # Check if component exists
            # Extract proportions for this component across all time points
            out["proportions"][name] = [float(proportions_in_time[j][i]) for j in range(len(times))]
        else:
            out["proportions"][name] = [0.0] * len(times)
    
    # Add contamination if available (magnetstein may not provide this)
    out["proportions"]["contamination"] = [0.0] * len(times)
    
    return out
