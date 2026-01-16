#!/usr/bin/env python3
"""
Benchmark script for reaction product prediction accuracy.

Tests ReactionT5 model against known organic reactions and evaluates:
- Top-1 / Top-5 accuracy
- SMILES validity
- Mean Reciprocal Rank (MRR)

Usage:
    python -m app.eval.benchmark_reactions
    python -m app.eval.benchmark_reactions --quick
    python -m app.eval.benchmark_reactions --output results.json
"""

import os
import sys
import json
import argparse
import time
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from datetime import datetime

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit import DataStructs

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


# ============================================================================
# Benchmark Reactions Dataset
# ============================================================================

BENCHMARK_REACTIONS = [
    # -------------------------------------------------------------------------
    # Electrophilic Aromatic Substitution
    # -------------------------------------------------------------------------
    {
        "id": "eas_bromination_anisole",
        "name": "Bromination of anisole",
        "category": "EAS",
        "reactants": "COc1ccccc1 . BrBr",
        "reagents": "Br[Fe](Br)Br",
        "expected_major": ["COc1ccc(Br)cc1"],  # para-bromoanisole
        "expected_minor": ["COc1ccccc1Br"],    # ortho-bromoanisole
        "notes": "Para-substitution favored due to steric and electronic effects",
    },
    {
        "id": "eas_bromination_toluene",
        "name": "Bromination of toluene",
        "category": "EAS",
        "reactants": "Cc1ccccc1 . BrBr",
        "reagents": "Br[Fe](Br)Br",
        "expected_major": ["Cc1ccc(Br)cc1"],   # para
        "expected_minor": ["Cc1ccccc1Br"],     # ortho
        "notes": "Methyl is ortho/para director",
    },
    {
        "id": "eas_bromination_benzene",
        "name": "Bromination of benzene",
        "category": "EAS",
        "reactants": "c1ccccc1 . BrBr",
        "reagents": "Br[Fe](Br)Br",
        "expected_major": ["Brc1ccccc1"],
        "expected_minor": [],
        "notes": "Simple substitution",
    },
    {
        "id": "eas_nitration_benzene",
        "name": "Nitration of benzene",
        "category": "EAS",
        "reactants": "c1ccccc1",
        "reagents": "O=[N+]([O-])O.OS(=O)(=O)O",  # HNO3 + H2SO4
        "expected_major": ["O=[N+]([O-])c1ccccc1"],
        "expected_minor": [],
        "notes": "Classic nitration",
    },

    # -------------------------------------------------------------------------
    # Esterification
    # -------------------------------------------------------------------------
    {
        "id": "ester_fischer_ethyl_acetate",
        "name": "Fischer esterification - ethyl acetate",
        "category": "Esterification",
        "reactants": "CC(=O)O . CCO",
        "reagents": "OS(=O)(=O)O",  # H2SO4
        "expected_major": ["CCOC(C)=O"],  # ethyl acetate
        "expected_minor": [],
        "notes": "Reversible, acid-catalyzed",
    },
    {
        "id": "ester_fischer_methyl_benzoate",
        "name": "Fischer esterification - methyl benzoate",
        "category": "Esterification",
        "reactants": "O=C(O)c1ccccc1 . CO",
        "reagents": "OS(=O)(=O)O",
        "expected_major": ["COC(=O)c1ccccc1"],
        "expected_minor": [],
        "notes": "Benzoic acid + methanol",
    },

    # -------------------------------------------------------------------------
    # Reduction
    # -------------------------------------------------------------------------
    {
        "id": "red_ketone_nabh4",
        "name": "Ketone reduction with NaBH4",
        "category": "Reduction",
        "reactants": "CC(=O)c1ccccc1",  # acetophenone
        "reagents": "[Na+].[BH4-]",
        "expected_major": ["CC(O)c1ccccc1"],  # 1-phenylethanol
        "expected_minor": [],
        "notes": "Selective ketone reduction",
    },
    {
        "id": "red_aldehyde_nabh4",
        "name": "Aldehyde reduction with NaBH4",
        "category": "Reduction",
        "reactants": "O=Cc1ccccc1",  # benzaldehyde
        "reagents": "[Na+].[BH4-]",
        "expected_major": ["OCc1ccccc1"],  # benzyl alcohol
        "expected_minor": [],
        "notes": "Aldehyde to primary alcohol",
    },

    # -------------------------------------------------------------------------
    # Grignard / Organometallic
    # -------------------------------------------------------------------------
    {
        "id": "grignard_formaldehyde",
        "name": "Grignard with formaldehyde",
        "category": "Grignard",
        "reactants": "[Mg+]c1ccccc1.[Br-] . C=O",  # PhMgBr + HCHO
        "reagents": "",
        "expected_major": ["OCc1ccccc1"],  # benzyl alcohol
        "expected_minor": [],
        "notes": "Primary alcohol formation",
    },

    # -------------------------------------------------------------------------
    # Aldol / Condensation
    # -------------------------------------------------------------------------
    {
        "id": "aldol_acetaldehyde",
        "name": "Aldol condensation of acetaldehyde",
        "category": "Aldol",
        "reactants": "CC=O . CC=O",
        "reagents": "[OH-]",
        "expected_major": ["CC(O)CC=O"],  # aldol product
        "expected_minor": ["CC=CC=O"],    # crotonaldehyde (dehydrated)
        "notes": "Base-catalyzed aldol",
    },

    # -------------------------------------------------------------------------
    # Diels-Alder
    # -------------------------------------------------------------------------
    {
        "id": "diels_alder_butadiene_ethylene",
        "name": "Diels-Alder: butadiene + ethylene",
        "category": "Cycloaddition",
        "reactants": "C=CC=C . C=C",
        "reagents": "",
        "expected_major": ["C1=CCCCC1"],  # cyclohexene
        "expected_minor": [],
        "notes": "4+2 cycloaddition",
    },
]

# Quick benchmark uses subset
QUICK_REACTION_IDS = [
    "eas_bromination_anisole",
    "ester_fischer_ethyl_acetate",
    "red_ketone_nabh4",
]


# ============================================================================
# SMILES Utilities
# ============================================================================

def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Return canonical SMILES or None if invalid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
    except:
        pass
    return None


def smiles_match(pred: str, expected: str) -> bool:
    """Check if two SMILES represent the same molecule."""
    can_pred = canonicalize_smiles(pred)
    can_exp = canonicalize_smiles(expected)
    if can_pred and can_exp:
        return can_pred == can_exp
    return False


def get_tanimoto_similarity(smiles1: str, smiles2: str) -> float:
    """Calculate Tanimoto similarity between two molecules."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 and mol2:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            return DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        pass
    return 0.0


def extract_components(smiles: str) -> List[str]:
    """Split multi-component SMILES and canonicalize each."""
    components = []
    for part in smiles.split('.'):
        part = part.strip()
        if part:
            can = canonicalize_smiles(part)
            if can:
                components.append(can)
    return components


# ============================================================================
# Metrics
# ============================================================================

@dataclass
class ReactionMetrics:
    """Metrics for a single reaction prediction."""
    reaction_id: str
    reaction_name: str
    category: str
    top1_correct: bool
    top5_correct: bool
    any_major_found: bool
    any_minor_found: bool
    rank_of_major: int  # -1 if not found
    n_predictions: int
    n_valid_predictions: int
    best_similarity: float
    runtime_seconds: float
    predictions: List[str]
    expected_major: List[str]
    expected_minor: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def evaluate_predictions(
    predictions: List[Tuple[str, float]],
    expected_major: List[str],
    expected_minor: List[str],
) -> Tuple[bool, bool, bool, bool, int, int, int, float]:
    """
    Evaluate prediction quality.

    Returns:
        (top1_correct, top5_correct, any_major, any_minor,
         rank_of_major, n_preds, n_valid, best_similarity)
    """
    # Canonicalize expected
    expected_major_can = set(filter(None, [canonicalize_smiles(s) for s in expected_major]))
    expected_minor_can = set(filter(None, [canonicalize_smiles(s) for s in expected_minor]))
    all_expected = expected_major_can | expected_minor_can

    # Process predictions
    pred_smiles = []
    valid_count = 0
    for pred, score in predictions:
        # Handle multi-component predictions
        components = extract_components(pred)
        pred_smiles.extend(components)
        if components:
            valid_count += 1

    # Remove duplicates while preserving order
    seen = set()
    unique_preds = []
    for s in pred_smiles:
        if s not in seen:
            seen.add(s)
            unique_preds.append(s)

    # Metrics
    n_preds = len(predictions)
    n_valid = valid_count

    # Top-1 / Top-5 accuracy
    top1_correct = len(unique_preds) > 0 and unique_preds[0] in expected_major_can
    top5_correct = any(s in expected_major_can for s in unique_preds[:5])

    # Did we find any expected products?
    any_major = any(s in expected_major_can for s in unique_preds)
    any_minor = any(s in expected_minor_can for s in unique_preds)

    # Rank of first major product found
    rank_of_major = -1
    for i, s in enumerate(unique_preds):
        if s in expected_major_can:
            rank_of_major = i + 1
            break

    # Best Tanimoto similarity to expected
    best_sim = 0.0
    for pred in unique_preds[:5]:
        for exp in all_expected:
            sim = get_tanimoto_similarity(pred, exp)
            best_sim = max(best_sim, sim)

    return (top1_correct, top5_correct, any_major, any_minor,
            rank_of_major, n_preds, n_valid, best_sim)


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_single_reaction(reaction: Dict) -> ReactionMetrics:
    """Run prediction on a single reaction and evaluate."""
    from app.tools_reactiont5 import propose_products

    start = time.time()

    try:
        predictions = propose_products(
            reactants=reaction["reactants"],
            reagents=reaction.get("reagents", ""),
            beams=10,
            n_best=5,
        )
    except Exception as e:
        print(f"  Error: {e}")
        predictions = []

    elapsed = time.time() - start

    # Evaluate
    (top1, top5, any_major, any_minor, rank, n_preds, n_valid, best_sim) = evaluate_predictions(
        predictions,
        reaction.get("expected_major", []),
        reaction.get("expected_minor", []),
    )

    # Extract just SMILES from predictions
    pred_smiles = [p[0] for p in predictions]

    metrics = ReactionMetrics(
        reaction_id=reaction["id"],
        reaction_name=reaction["name"],
        category=reaction["category"],
        top1_correct=top1,
        top5_correct=top5,
        any_major_found=any_major,
        any_minor_found=any_minor,
        rank_of_major=rank,
        n_predictions=n_preds,
        n_valid_predictions=n_valid,
        best_similarity=best_sim,
        runtime_seconds=elapsed,
        predictions=pred_smiles,
        expected_major=reaction.get("expected_major", []),
        expected_minor=reaction.get("expected_minor", []),
    )

    return metrics


def run_full_benchmark(
    quick: bool = False,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Run the complete reaction prediction benchmark.

    Args:
        quick: If True, run only subset of reactions
        output_path: Optional path to save JSON results

    Returns:
        Dict with benchmark results
    """
    print("=" * 60)
    print("MixSense Reaction Prediction Benchmark")
    print("=" * 60)

    # Select reactions
    if quick:
        reactions = [r for r in BENCHMARK_REACTIONS if r["id"] in QUICK_REACTION_IDS]
    else:
        reactions = BENCHMARK_REACTIONS

    print(f"\nRunning {len(reactions)} reactions...")

    all_results = []
    categories: Dict[str, List[ReactionMetrics]] = {}

    for reaction in reactions:
        print(f"\n--- {reaction['name']} ---")
        print(f"  Reactants: {reaction['reactants']}")
        print(f"  Reagents: {reaction.get('reagents', 'none')}")
        print(f"  Expected: {reaction.get('expected_major', [])}")

        metrics = run_single_reaction(reaction)
        all_results.append(metrics)

        # Track by category
        cat = reaction["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(metrics)

        # Report
        status = "PASS" if metrics.top1_correct else ("TOP5" if metrics.top5_correct else "FAIL")
        print(f"  Predictions: {metrics.predictions[:3]}")
        print(f"  Result: {status} | Rank={metrics.rank_of_major} | Sim={metrics.best_similarity:.3f}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_total = len(all_results)
    n_top1 = sum(1 for r in all_results if r.top1_correct)
    n_top5 = sum(1 for r in all_results if r.top5_correct)
    n_any_major = sum(1 for r in all_results if r.any_major_found)

    print(f"\nOverall ({n_total} reactions):")
    print(f"  Top-1 Accuracy: {n_top1}/{n_total} = {100*n_top1/n_total:.1f}%")
    print(f"  Top-5 Accuracy: {n_top5}/{n_total} = {100*n_top5/n_total:.1f}%")
    print(f"  Any Major Found: {n_any_major}/{n_total} = {100*n_any_major/n_total:.1f}%")

    # MRR (Mean Reciprocal Rank)
    ranks = [r.rank_of_major for r in all_results if r.rank_of_major > 0]
    mrr = sum(1.0/r for r in ranks) / len(all_results) if all_results else 0.0
    print(f"  MRR: {mrr:.3f}")

    # Per-category breakdown
    print("\nBy Category:")
    for cat, results in sorted(categories.items()):
        n = len(results)
        t1 = sum(1 for r in results if r.top1_correct)
        t5 = sum(1 for r in results if r.top5_correct)
        print(f"  {cat}: Top-1={t1}/{n} ({100*t1/n:.0f}%), Top-5={t5}/{n} ({100*t5/n:.0f}%)")

    # Prepare output
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "quick": quick,
            "n_reactions": len(reactions),
        },
        "results": [r.to_dict() for r in all_results],
        "summary": {
            "total": n_total,
            "top1_accuracy": n_top1 / n_total if n_total > 0 else 0,
            "top5_accuracy": n_top5 / n_total if n_total > 0 else 0,
            "any_major_rate": n_any_major / n_total if n_total > 0 else 0,
            "mrr": mrr,
            "by_category": {
                cat: {
                    "n": len(results),
                    "top1": sum(1 for r in results if r.top1_correct) / len(results),
                    "top5": sum(1 for r in results if r.top5_correct) / len(results),
                }
                for cat, results in categories.items()
            }
        },
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
    parser = argparse.ArgumentParser(description="Benchmark reaction product prediction")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark (fewer reactions)")
    parser.add_argument("--output", "-o", type=str, help="Output JSON file path")

    args = parser.parse_args()

    run_full_benchmark(
        quick=args.quick,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
