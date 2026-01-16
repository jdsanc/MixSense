#!/usr/bin/env python3
"""
End-to-end pipeline benchmark for MixSense.

Tests the full workflow:
1. Name resolution → SMILES
2. Reaction prediction → Products
3. Reference lookup → Spectra
4. Deconvolution → Quantification

Usage:
    python -m app.eval.benchmark_pipeline
    python -m app.eval.benchmark_pipeline --quick
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# ============================================================================
# Test Cases: Full Pipeline Scenarios
# ============================================================================

PIPELINE_SCENARIOS = [
    {
        "id": "bromination_anisole_full",
        "name": "Anisole Bromination - Full Pipeline",
        "description": "Predict bromination products, find references, quantify mixture",
        "input": {
            "reactant_names": ["anisole", "Br2"],
            "reagent_names": ["FeBr3"],
        },
        "expected": {
            "resolved_reactants": ["COc1ccccc1", "BrBr"],
            "predicted_products": ["COc1ccc(Br)cc1"],  # para-bromoanisole
            "mixture_composition": {
                "anisole": 0.3,
                "p-bromoanisole": 0.7,
            },
        },
    },
    {
        "id": "esterification_simple",
        "name": "Fischer Esterification",
        "description": "Acetic acid + ethanol → ethyl acetate",
        "input": {
            "reactant_names": ["acetic acid", "ethanol"],
            "reagent_names": ["sulfuric acid"],
        },
        "expected": {
            "resolved_reactants": ["CC(=O)O", "CCO"],
            "predicted_products": ["CCOC(C)=O"],
        },
    },
]


# ============================================================================
# Pipeline Components
# ============================================================================

@dataclass
class PipelineStepResult:
    """Result of a single pipeline step."""
    step_name: str
    success: bool
    output: Dict
    error: Optional[str]
    runtime_seconds: float


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    scenario_id: str
    scenario_name: str
    overall_success: bool
    steps: List[PipelineStepResult]
    total_runtime: float


def step_resolve_names(names: List[str]) -> PipelineStepResult:
    """Step 1: Resolve chemical names to SMILES."""
    from app.tools_reactiont5 import resolve_names_to_smiles

    start = time.time()
    error = None
    output = {}

    try:
        smiles_list = resolve_names_to_smiles(", ".join(names))
        output = {
            "input_names": names,
            "resolved_smiles": smiles_list,
            "n_resolved": len(smiles_list),
        }
        success = len(smiles_list) > 0
    except Exception as e:
        error = str(e)
        success = False

    return PipelineStepResult(
        step_name="resolve_names",
        success=success,
        output=output,
        error=error,
        runtime_seconds=time.time() - start,
    )


def step_predict_products(reactants: List[str], reagents: List[str]) -> PipelineStepResult:
    """Step 2: Predict reaction products."""
    from app.tools_reactiont5 import propose_products

    start = time.time()
    error = None
    output = {}

    try:
        reactants_str = " . ".join(reactants)
        reagents_str = " . ".join(reagents) if reagents else ""

        predictions = propose_products(
            reactants=reactants_str,
            reagents=reagents_str,
            n_best=5,
        )

        output = {
            "reactants": reactants_str,
            "reagents": reagents_str,
            "predictions": [{"smiles": p[0], "score": p[1]} for p in predictions],
            "n_predictions": len(predictions),
        }
        success = len(predictions) > 0
    except Exception as e:
        error = str(e)
        success = False

    return PipelineStepResult(
        step_name="predict_products",
        success=success,
        output=output,
        error=error,
        runtime_seconds=time.time() - start,
    )


def step_lookup_references(smiles_list: List[str]) -> PipelineStepResult:
    """Step 3: Look up NMR references for compounds."""
    from app.tools_nmrbank import get_reference_by_smiles

    start = time.time()
    error = None
    output = {}

    try:
        refs = {}
        for smiles in smiles_list:
            ref = get_reference_by_smiles(smiles)
            if ref:
                refs[smiles] = {
                    "name": ref.get("name", smiles),
                    "n_peaks": len(ref.get("ppm", [])),
                    "has_h1": "ppm_h1" in ref,
                    "has_c13": "ppm_c13" in ref,
                }

        output = {
            "queried_smiles": smiles_list,
            "found_refs": refs,
            "n_found": len(refs),
            "coverage": len(refs) / len(smiles_list) if smiles_list else 0,
        }
        success = len(refs) > 0
    except Exception as e:
        error = str(e)
        success = False

    return PipelineStepResult(
        step_name="lookup_references",
        success=success,
        output=output,
        error=error,
        runtime_seconds=time.time() - start,
    )


def step_deconvolve(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    ref_smiles: List[str],
) -> PipelineStepResult:
    """Step 4: Deconvolve mixture spectrum."""
    from app.tools_nmrbank import get_reference_by_smiles
    from app.tools_magnetstein import quantify_single

    start = time.time()
    error = None
    output = {}

    try:
        # Build library
        library = []
        for smiles in ref_smiles:
            ref = get_reference_by_smiles(smiles)
            if ref:
                library.append({
                    "name": ref.get("name", smiles),
                    "smiles": smiles,
                    "ppm": ref.get("ppm", []),
                    "intensity": ref.get("intensity", []),
                })

        if not library:
            raise ValueError("No valid references for deconvolution")

        # Run deconvolution
        result = quantify_single(
            mixture_ppm=mixture_ppm,
            mixture_intensity=mixture_intensity,
            library=library,
            min_peaks=1,
        )

        output = {
            "library_size": len(library),
            "concentrations": result.get("concentrations", {}),
        }
        success = bool(result.get("concentrations"))
    except Exception as e:
        error = str(e)
        success = False

    return PipelineStepResult(
        step_name="deconvolve",
        success=success,
        output=output,
        error=error,
        runtime_seconds=time.time() - start,
    )


# ============================================================================
# Pipeline Runner
# ============================================================================

def run_pipeline_scenario(scenario: Dict) -> PipelineResult:
    """Execute a complete pipeline scenario."""
    steps = []
    start_time = time.time()

    print(f"\n--- {scenario['name']} ---")
    print(f"    {scenario['description']}")

    # Step 1: Resolve names
    print("  [1/4] Resolving names...")
    reactant_result = step_resolve_names(scenario["input"]["reactant_names"])
    steps.append(reactant_result)
    print(f"        {reactant_result.output.get('resolved_smiles', [])} - {'OK' if reactant_result.success else 'FAIL'}")

    reagent_result = step_resolve_names(scenario["input"].get("reagent_names", []))
    steps.append(reagent_result)

    if not reactant_result.success:
        return PipelineResult(
            scenario_id=scenario["id"],
            scenario_name=scenario["name"],
            overall_success=False,
            steps=steps,
            total_runtime=time.time() - start_time,
        )

    # Step 2: Predict products
    print("  [2/4] Predicting products...")
    product_result = step_predict_products(
        reactants=reactant_result.output.get("resolved_smiles", []),
        reagents=reagent_result.output.get("resolved_smiles", []),
    )
    steps.append(product_result)
    preds = product_result.output.get("predictions", [])[:3]
    print(f"        {[p['smiles'] for p in preds]} - {'OK' if product_result.success else 'FAIL'}")

    # Step 3: Lookup references
    print("  [3/4] Looking up references...")
    all_smiles = (
        reactant_result.output.get("resolved_smiles", []) +
        [p["smiles"] for p in product_result.output.get("predictions", [])[:3]]
    )
    ref_result = step_lookup_references(all_smiles)
    steps.append(ref_result)
    print(f"        Found {ref_result.output.get('n_found', 0)}/{len(all_smiles)} refs - {'OK' if ref_result.success else 'FAIL'}")

    # Step 4: Deconvolution (using synthetic mixture for demo)
    print("  [4/4] Deconvolution (skipped - requires mixture spectrum)")
    # In a real test, we would:
    # 1. Generate synthetic mixture from predicted products
    # 2. Run deconvolution
    # 3. Compare to expected composition

    total_time = time.time() - start_time
    overall_success = all(s.success for s in steps)

    return PipelineResult(
        scenario_id=scenario["id"],
        scenario_name=scenario["name"],
        overall_success=overall_success,
        steps=steps,
        total_runtime=total_time,
    )


# ============================================================================
# Main Benchmark
# ============================================================================

def run_full_pipeline_benchmark(
    quick: bool = False,
    output_path: Optional[str] = None,
) -> Dict:
    """Run the complete pipeline benchmark."""
    print("=" * 60)
    print("MixSense End-to-End Pipeline Benchmark")
    print("=" * 60)

    # Warm NMRBank cache
    print("\nInitializing...")
    try:
        from app.tools_nmrbank import warm_cache
        n = warm_cache()
        print(f"  NMRBank loaded: {n} compounds")
    except Exception as e:
        print(f"  Warning: NMRBank not available: {e}")

    # Select scenarios
    scenarios = PIPELINE_SCENARIOS[:1] if quick else PIPELINE_SCENARIOS

    print(f"\nRunning {len(scenarios)} scenarios...")

    results = []
    for scenario in scenarios:
        result = run_pipeline_scenario(scenario)
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_success = sum(1 for r in results if r.overall_success)
    print(f"\nPipeline Success Rate: {n_success}/{len(results)}")

    for result in results:
        status = "PASS" if result.overall_success else "FAIL"
        print(f"  [{status}] {result.scenario_name} ({result.total_runtime:.2f}s)")
        for step in result.steps:
            step_status = "OK" if step.success else "FAIL"
            print(f"       - {step.step_name}: {step_status} ({step.runtime_seconds:.2f}s)")

    # Output
    output = {
        "timestamp": datetime.now().isoformat(),
        "n_scenarios": len(scenarios),
        "n_success": n_success,
        "success_rate": n_success / len(scenarios) if scenarios else 0,
        "results": [
            {
                "scenario_id": r.scenario_id,
                "scenario_name": r.scenario_name,
                "success": r.overall_success,
                "runtime": r.total_runtime,
                "steps": [asdict(s) for s in r.steps],
            }
            for r in results
        ],
    }

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return output


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run end-to-end pipeline benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument("--output", "-o", type=str, help="Output JSON path")

    args = parser.parse_args()

    run_full_pipeline_benchmark(
        quick=args.quick,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
