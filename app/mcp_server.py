#!/usr/bin/env python3
"""
MCP Server for NMR Chemistry Tools.

Exposes chemistry analysis tools via Model Context Protocol:
- lookup_nmr_reference: Find NMR reference spectra by SMILES
- predict_products: Predict reaction products using ReactionT5
- deconvolve_mixture: Deconvolve NMR mixture into components

Run with: python -m app.mcp_server
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mcp.server.fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("nmr-chemistry")


@mcp.tool()
def lookup_nmr_reference(smiles: str) -> Dict[str, Any]:
    """
    Look up NMR reference spectrum for a compound by SMILES.

    Args:
        smiles: SMILES string of the compound to look up

    Returns:
        Dictionary with name, smiles, ppm (chemical shifts), and intensity values.
        Returns empty dict if not found.
    """
    from app.tools_nmrbank import get_reference_by_smiles

    result = get_reference_by_smiles(smiles)
    if result:
        return {
            "name": result.get("name", ""),
            "smiles": result.get("smiles", smiles),
            "ppm": result.get("ppm", []),
            "intensity": result.get("intensity", []),
        }
    return {"error": f"No reference found for SMILES: {smiles}"}


@mcp.tool()
def resolve_chemical_name(name: str) -> Dict[str, Any]:
    """
    Convert a chemical name to canonical SMILES.

    Args:
        name: Chemical name (e.g., "anisole", "benzene", "alpha-pinene")

    Returns:
        Dictionary with the canonical SMILES string, or error if not found.
    """
    from app.tools_reactiont5 import resolve_names_to_smiles

    smiles_list = resolve_names_to_smiles(name)
    if smiles_list:
        return {"smiles": smiles_list[0], "all_matches": smiles_list}
    return {"error": f"Could not resolve name: {name}"}


@mcp.tool()
def predict_products(
    reactants: str,
    reagents: str = "",
    n_best: int = 5
) -> Dict[str, Any]:
    """
    Predict reaction products using ReactionT5 model.

    Args:
        reactants: Reactant SMILES separated by " . " (e.g., "COc1ccccc1 . BrBr")
        reagents: Optional reagents/conditions (e.g., "FeBr3")
        n_best: Number of top predictions to return (default: 5)

    Returns:
        Dictionary with predicted product SMILES and scores.
    """
    from app.tools_reactiont5 import propose_products, get_unique_components

    predictions = propose_products(reactants, reagents=reagents, n_best=n_best)
    products = [smiles for smiles, score in predictions]
    unique = get_unique_components(products)

    return {
        "predictions": [{"smiles": s, "score": sc} for s, sc in predictions],
        "unique_components": unique,
    }


@mcp.tool()
def deconvolve_mixture(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    references: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Deconvolve an NMR mixture spectrum into component contributions.

    Args:
        mixture_ppm: PPM values of mixture spectrum
        mixture_intensity: Intensity values of mixture spectrum
        references: List of reference spectra, each with keys: name, ppm, intensity

    Returns:
        Dictionary with component concentrations.
    """
    from app.tools_magnetstein import quantify_single

    if not mixture_ppm or not mixture_intensity:
        return {"error": "Empty mixture data"}

    if not references:
        return {"error": "No reference spectra provided"}

    result = quantify_single(
        mixture_ppm=mixture_ppm,
        mixture_intensity=mixture_intensity,
        library=references,
    )

    return {
        "concentrations": result.get("concentrations", {}),
    }


@mcp.tool()
def load_references_for_smiles(smiles_list: List[str]) -> Dict[str, Any]:
    """
    Load NMR reference spectra for a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings to look up

    Returns:
        Dictionary with found references and any missing compounds.
    """
    from app.tools_nmrbank import get_reference_by_smiles

    found = []
    missing = []

    for smiles in smiles_list:
        ref = get_reference_by_smiles(smiles)
        if ref:
            found.append({
                "name": ref.get("name", ""),
                "smiles": ref.get("smiles", smiles),
                "ppm": ref.get("ppm", []),
                "intensity": ref.get("intensity", []),
            })
        else:
            missing.append(smiles)

    return {"references": found, "missing": missing}


def main():
    """Run the MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="NMR Chemistry MCP Server")
    parser.add_argument("--transport", choices=["stdio", "sse"], default="stdio",
                        help="Transport method (default: stdio)")
    args = parser.parse_args()

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    else:
        mcp.run(transport="sse")


if __name__ == "__main__":
    main()
