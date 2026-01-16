#!/usr/bin/env python3
"""
Fallback NMR spectrum prediction when NMRBank doesn't have a compound.

Methods (in order of preference):
1. NMRBank lookup (existing)
2. Group contribution / HOSE-code based prediction
3. Structural analog search

Usage:
    from app.eval.predict_nmr import get_reference_with_fallback

    ref, source = get_reference_with_fallback("COc1ccc(Br)cc1")
    # source is "nmrbank", "predicted", or "analog"
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


# ============================================================================
# Group Contribution NMR Prediction
# ============================================================================

# Approximate 1H chemical shift ranges for common functional groups
# Based on standard NMR correlation tables
H1_SHIFT_TABLE = {
    # Alkyl groups
    "CH3_alkyl": (0.8, 1.0),          # R-CH3
    "CH2_alkyl": (1.2, 1.4),          # R-CH2-R
    "CH_alkyl": (1.4, 1.6),           # R3CH

    # Adjacent to electronegative groups
    "CH3_OR": (3.3, 3.5),             # CH3-O-R
    "CH2_OR": (3.4, 3.6),             # R-CH2-O-R
    "CH3_OAr": (3.7, 3.9),            # CH3-O-Ar (methoxy)
    "CH2_OH": (3.5, 3.7),             # R-CH2-OH
    "CH_OH": (3.8, 4.0),              # R2CH-OH

    # Adjacent to carbonyl
    "CH3_CO": (2.0, 2.2),             # CH3-C=O
    "CH2_CO": (2.3, 2.5),             # R-CH2-C=O
    "CH_aldehyde": (9.5, 10.0),       # R-CHO

    # Adjacent to halogens
    "CH2_Cl": (3.4, 3.6),
    "CH2_Br": (3.3, 3.5),
    "CH_Cl": (3.8, 4.1),
    "CH_Br": (4.1, 4.4),

    # Aromatics
    "ArH_benzene": (7.2, 7.4),        # Unsubstituted benzene
    "ArH_electron_rich": (6.7, 7.0),  # Ortho/para to -OR, -OH
    "ArH_electron_poor": (7.5, 8.0),  # Ortho/para to -NO2, -COR
    "ArH_halogenated": (7.1, 7.5),    # Ortho/para to halogen

    # Heteroaromatics
    "pyridine_2": (8.4, 8.6),
    "pyridine_3": (7.2, 7.4),
    "pyridine_4": (7.5, 7.7),

    # Alkenes
    "CH_vinyl": (5.0, 5.5),
    "CH2_vinyl": (4.8, 5.2),
}


def predict_aromatic_shifts(mol) -> List[Tuple[float, float]]:
    """
    Predict 1H shifts for aromatic protons based on substituent effects.

    Returns list of (ppm, intensity) tuples.
    """
    peaks = []

    # Find aromatic rings
    ring_info = mol.GetRingInfo()
    aromatic_atoms = set()

    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            aromatic_atoms.add(atom.GetIdx())

    # For each aromatic H, estimate shift
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in aromatic_atoms:
            continue
        if atom.GetAtomicNum() != 6:  # Only carbon
            continue

        n_h = atom.GetTotalNumHs()
        if n_h == 0:
            continue

        # Check substituent effects
        neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]
        has_electron_donor = False
        has_electron_withdrawer = False
        has_halogen = False

        for ring_neighbor in neighbors:
            if ring_neighbor.GetIdx() in aromatic_atoms:
                # Check substituents on ring
                for sub in ring_neighbor.GetNeighbors():
                    if sub.GetIdx() not in aromatic_atoms:
                        sym = sub.GetSymbol()
                        if sym == "O" or sym == "N":
                            has_electron_donor = True
                        elif sym in ["F", "Cl", "Br", "I"]:
                            has_halogen = True
                        elif sym == "C":
                            # Check for carbonyl
                            for nn in sub.GetNeighbors():
                                if nn.GetSymbol() == "O" and sub.GetBondWith(nn).GetBondTypeAsDouble() == 2.0:
                                    has_electron_withdrawer = True

        # Assign shift based on electronic effects
        if has_electron_donor:
            shift_range = H1_SHIFT_TABLE["ArH_electron_rich"]
        elif has_electron_withdrawer:
            shift_range = H1_SHIFT_TABLE["ArH_electron_poor"]
        elif has_halogen:
            shift_range = H1_SHIFT_TABLE["ArH_halogenated"]
        else:
            shift_range = H1_SHIFT_TABLE["ArH_benzene"]

        # Use midpoint
        shift = (shift_range[0] + shift_range[1]) / 2
        peaks.append((shift, float(n_h)))

    return peaks


def predict_aliphatic_shifts(mol) -> List[Tuple[float, float]]:
    """
    Predict 1H shifts for aliphatic protons.

    Returns list of (ppm, intensity) tuples.
    """
    peaks = []

    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 6:  # Only carbon
            continue
        if atom.GetIsAromatic():
            continue

        n_h = atom.GetTotalNumHs()
        if n_h == 0:
            continue

        # Determine environment
        neighbors = [mol.GetAtomWithIdx(n.GetIdx()) for n in atom.GetNeighbors()]
        neighbor_symbols = [n.GetSymbol() for n in neighbors]

        # Check for special environments
        shift_key = None

        if "O" in neighbor_symbols:
            # Adjacent to oxygen
            for n in neighbors:
                if n.GetSymbol() == "O":
                    # Check if methoxy attached to aromatic
                    o_neighbors = [mol.GetAtomWithIdx(nn.GetIdx()) for nn in n.GetNeighbors()]
                    if any(nn.GetIsAromatic() for nn in o_neighbors):
                        shift_key = "CH3_OAr" if n_h == 3 else "CH2_OR"
                    else:
                        shift_key = "CH3_OR" if n_h == 3 else "CH2_OR"
                    break

        elif "Cl" in neighbor_symbols:
            shift_key = "CH2_Cl" if n_h >= 2 else "CH_Cl"
        elif "Br" in neighbor_symbols:
            shift_key = "CH2_Br" if n_h >= 2 else "CH_Br"

        else:
            # Check for adjacent carbonyl
            for n in neighbors:
                if n.GetSymbol() == "C":
                    for nn in n.GetNeighbors():
                        if nn.GetSymbol() == "O":
                            bond = n.GetBondWith(nn)
                            if bond and bond.GetBondTypeAsDouble() == 2.0:
                                shift_key = "CH3_CO" if n_h == 3 else "CH2_CO"
                                break

        if shift_key is None:
            # Default aliphatic
            if n_h == 3:
                shift_key = "CH3_alkyl"
            elif n_h == 2:
                shift_key = "CH2_alkyl"
            else:
                shift_key = "CH_alkyl"

        if shift_key in H1_SHIFT_TABLE:
            shift_range = H1_SHIFT_TABLE[shift_key]
            shift = (shift_range[0] + shift_range[1]) / 2
            peaks.append((shift, float(n_h)))

    return peaks


def predict_nmr_spectrum(smiles: str) -> Optional[Dict]:
    """
    Predict 1H NMR spectrum using group contribution method.

    Args:
        smiles: SMILES string

    Returns:
        Dict with {name, smiles, ppm, intensity} or None if failed
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol = Chem.AddHs(mol)

        peaks = []

        # Aromatic protons
        peaks.extend(predict_aromatic_shifts(mol))

        # Aliphatic protons
        peaks.extend(predict_aliphatic_shifts(mol))

        if not peaks:
            return None

        # Sort by chemical shift (descending, NMR convention)
        peaks.sort(key=lambda x: x[0], reverse=True)

        # Consolidate nearby peaks (within 0.1 ppm)
        consolidated = []
        for ppm, intensity in peaks:
            merged = False
            for i, (p, inten) in enumerate(consolidated):
                if abs(ppm - p) < 0.1:
                    # Merge
                    new_ppm = (p * inten + ppm * intensity) / (inten + intensity)
                    consolidated[i] = (new_ppm, inten + intensity)
                    merged = True
                    break
            if not merged:
                consolidated.append((ppm, intensity))

        ppm_list = [p for p, _ in consolidated]
        intensity_list = [i for _, i in consolidated]

        # Normalize intensity
        max_int = max(intensity_list) if intensity_list else 1.0
        intensity_list = [i / max_int for i in intensity_list]

        return {
            "name": f"Predicted ({smiles[:20]}...)" if len(smiles) > 20 else f"Predicted ({smiles})",
            "smiles": smiles,
            "ppm": ppm_list,
            "intensity": intensity_list,
            "source": "predicted",
            "method": "group_contribution",
        }

    except Exception as e:
        print(f"Prediction failed for {smiles}: {e}")
        return None


# ============================================================================
# Structural Analog Search
# ============================================================================

def get_structural_analog(
    smiles: str,
    available_smiles: List[str],
    threshold: float = 0.7,
) -> Optional[str]:
    """
    Find the most similar compound from available references.

    Args:
        smiles: Query SMILES
        available_smiles: List of available reference SMILES
        threshold: Minimum Tanimoto similarity

    Returns:
        Most similar SMILES or None
    """
    try:
        query_mol = Chem.MolFromSmiles(smiles)
        if query_mol is None:
            return None

        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)

        best_sim = 0.0
        best_smiles = None

        for ref_smiles in available_smiles:
            ref_mol = Chem.MolFromSmiles(ref_smiles)
            if ref_mol is None:
                continue

            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)

            from rdkit import DataStructs
            sim = DataStructs.TanimotoSimilarity(query_fp, ref_fp)

            if sim > best_sim:
                best_sim = sim
                best_smiles = ref_smiles

        if best_sim >= threshold:
            return best_smiles
        return None

    except Exception as e:
        print(f"Analog search failed for {smiles}: {e}")
        return None


# ============================================================================
# Main Interface
# ============================================================================

def get_reference_with_fallback(
    smiles: str,
    use_prediction: bool = True,
    use_analog: bool = True,
    available_refs: Optional[Dict[str, Dict]] = None,
) -> Tuple[Optional[Dict], str]:
    """
    Get NMR reference spectrum with fallback to prediction/analog.

    Args:
        smiles: Query SMILES
        use_prediction: Whether to use group contribution prediction
        use_analog: Whether to search for structural analogs
        available_refs: Dict of {smiles: spectrum} for analog search

    Returns:
        (spectrum_dict, source) where source is "nmrbank", "predicted", "analog", or "none"
    """
    # 1. Try NMRBank first
    try:
        from app.tools_nmrbank import get_reference_by_smiles
        ref = get_reference_by_smiles(smiles)
        if ref:
            ref["source"] = "nmrbank"
            return ref, "nmrbank"
    except ImportError:
        pass

    # 2. Try prediction
    if use_prediction:
        ref = predict_nmr_spectrum(smiles)
        if ref:
            return ref, "predicted"

    # 3. Try analog search
    if use_analog and available_refs:
        analog_smiles = get_structural_analog(
            smiles,
            list(available_refs.keys()),
            threshold=0.7
        )
        if analog_smiles:
            ref = available_refs[analog_smiles].copy()
            ref["source"] = "analog"
            ref["analog_of"] = analog_smiles
            return ref, "analog"

    return None, "none"


# ============================================================================
# Demo / Testing
# ============================================================================

if __name__ == "__main__":
    print("NMR Spectrum Prediction Demo")
    print("=" * 50)

    test_cases = [
        ("COc1ccccc1", "anisole"),
        ("COc1ccc(Br)cc1", "4-bromoanisole"),
        ("Cc1ccccc1", "toluene"),
        ("CC(=O)O", "acetic acid"),
        ("CCOC(C)=O", "ethyl acetate"),
        ("c1ccccc1Br", "bromobenzene"),
    ]

    for smiles, name in test_cases:
        print(f"\n{name} ({smiles}):")
        ref = predict_nmr_spectrum(smiles)
        if ref:
            print(f"  Peaks: {len(ref['ppm'])}")
            for ppm, intensity in zip(ref['ppm'], ref['intensity']):
                print(f"    {ppm:.2f} ppm (intensity {intensity:.2f})")
        else:
            print("  Prediction failed")
