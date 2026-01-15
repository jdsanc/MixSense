#!/usr/bin/env python3
"""
Simple Chemistry Agent that uses MCP tools directly.

This is a straightforward implementation that:
1. Takes a natural language chemistry request
2. Uses tools to analyze the request
3. Returns structured results

Can be used standalone or as a reference for building more complex agents.
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.tools_nmrbank import get_reference_by_smiles
from app.tools_reactiont5 import resolve_names_to_smiles, propose_products, get_unique_components
from app.tools_deconvolve import deconvolve_spectra


def analyze_reaction(
    reactants: List[str],
    reagents: str = "",
    mixture_ppm: Optional[List[float]] = None,
    mixture_intensity: Optional[List[float]] = None,
    n_best: int = 5
) -> Dict[str, Any]:
    """
    Analyze a chemical reaction: predict products, find references, and optionally quantify.

    Args:
        reactants: List of reactant names or SMILES
        reagents: Optional reagent/catalyst string
        mixture_ppm: Optional mixture spectrum PPM values for quantification
        mixture_intensity: Optional mixture spectrum intensity values
        n_best: Number of product predictions to return

    Returns:
        Dictionary with analysis results
    """
    results = {
        "reactants": [],
        "products": [],
        "references": [],
        "quantification": None,
        "errors": []
    }

    # Step 1: Resolve reactant names to SMILES
    reactant_smiles = []
    for r in reactants:
        resolved = resolve_names_to_smiles(r)
        if resolved:
            reactant_smiles.extend(resolved)
            results["reactants"].append({"input": r, "smiles": resolved[0]})
        else:
            results["errors"].append(f"Could not resolve: {r}")

    if not reactant_smiles:
        results["errors"].append("No valid reactants found")
        return results

    # Step 2: Predict products
    reactants_str = " . ".join(reactant_smiles)
    predictions = propose_products(reactants_str, reagents=reagents, n_best=n_best)
    product_smiles = [s for s, _ in predictions]
    unique_products = get_unique_components(product_smiles)

    results["products"] = [{"smiles": s, "score": sc} for s, sc in predictions]

    # Step 3: Find reference spectra for all species
    all_species = reactant_smiles + unique_products
    for smiles in all_species:
        ref = get_reference_by_smiles(smiles)
        if ref:
            results["references"].append({
                "name": ref.get("name", ""),
                "smiles": smiles,
                "ppm": ref.get("ppm", []),
                "intensity": ref.get("intensity", [])
            })

    # Step 4: Quantify mixture if data provided
    if mixture_ppm and mixture_intensity and results["references"]:
        try:
            quant = deconvolve_spectra(
                mixture_ppm=mixture_ppm,
                mixture_intensity=mixture_intensity,
                refs=results["references"],
                quiet=True
            )
            results["quantification"] = quant.get("concentrations", {})
        except Exception as e:
            results["errors"].append(f"Quantification failed: {e}")

    return results


def main():
    """Demo the simple agent."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Simple Chemistry Agent")
    parser.add_argument("--reactants", nargs="+", required=True,
                        help="Reactant names or SMILES")
    parser.add_argument("--reagents", default="",
                        help="Reagents/conditions")
    parser.add_argument("--mixture-csv",
                        help="CSV file with mixture spectrum (ppm,intensity)")
    args = parser.parse_args()

    # Load mixture data if provided
    mixture_ppm = None
    mixture_intensity = None
    if args.mixture_csv:
        import pandas as pd
        df = pd.read_csv(args.mixture_csv, header=None, names=["ppm", "intensity"])
        mixture_ppm = df["ppm"].tolist()
        mixture_intensity = df["intensity"].tolist()

    # Run analysis
    results = analyze_reaction(
        reactants=args.reactants,
        reagents=args.reagents,
        mixture_ppm=mixture_ppm,
        mixture_intensity=mixture_intensity
    )

    # Print results
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
