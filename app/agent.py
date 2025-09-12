from typing import List, Dict, Any, Tuple
import pandas as pd
from rdkit import Chem
from .tools_reactiont5 import propose_products
from .tools_nmrbank import get_reference_by_smiles
from .tools_asics import asics_quantify

def normalize_smiles_list(text: str) -> List[str]:
    """
    Enhanced chemical name normalization using reactants_to_products utilities if available
    """
    # Try to use the enhanced normalization from reactants_to_products
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        from reactants_to_products.utils import normalize_chemical_names
        
        # Use the enhanced normalization function
        result = normalize_chemical_names(text)
        if result:
            print(f"Enhanced normalization: {text} -> {result}")
            return result
    except ImportError:
        print("Using fallback chemical name normalization")
    
    # Fallback to original implementation
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    out = []
    
    # Extended chemical name to SMILES mapping (enhanced from original)
    name_to_smiles = {
        "anisole": "COc1ccccc1",
        "benzene": "c1ccccc1",
        "toluene": "Cc1ccccc1",
        "phenol": "Oc1ccccc1",
        "bromobenzene": "Brc1ccccc1",
        "p-bromoanisole": "COc1ccc(Br)cc1",
        "o-bromoanisole": "COc1ccccc1Br",
        "m-bromoanisole": "COc1cccc(Br)c1",
        "br2": "BrBr",
        "bromine": "BrBr",
        "febr3": "Br[Fe](Br)Br",
        "iron(iii) bromide": "Br[Fe](Br)Br",
        # Additional common chemicals
        "acetic acid": "CC(=O)O",
        "ethanol": "CCO",
        "water": "O",
        "sulfuric acid": "OS(=O)(=O)O",
        "hydrochloric acid": "Cl",
        "methanol": "CO",
        "acetone": "CC(=O)C",
        "chloroform": "C(Cl)(Cl)Cl",
        "dichloromethane": "C(Cl)Cl",
        "diethyl ether": "CCOCC",
    }
    
    for p in parts:
        mol = None
        
        # First try to parse as SMILES directly
        try:
            mol = Chem.MolFromSmiles(p)
        except:
            mol = None
        
        if mol is None:
            # Try common name lookup (case insensitive)
            normalized_name = p.lower().strip()
            if normalized_name in name_to_smiles:
                try:
                    mol = Chem.MolFromSmiles(name_to_smiles[normalized_name])
                except:
                    mol = None
        
        if mol is None:
            print(f"Warning: Could not parse '{p}' as SMILES or chemical name")
            continue
            
        if mol:
            canonical = Chem.MolToSmiles(mol, canonical=True)
            out.append(canonical)
    
    return list(dict.fromkeys(out))

def step_propose(reactants_smiles: List[str], reagents: str, topk=5) -> List[str]:
    reactants_str = " . ".join(reactants_smiles)
    preds = propose_products(reactants_str, reagents=reagents, n_best=topk)
    return [s for s,_ in preds]

def load_refs_for_species(smiles_list: List[str]) -> List[Dict[str, Any]]:
    refs = []
    for s in smiles_list:
        ref = get_reference_by_smiles(s)
        if ref:
            refs.append({"name": ref["name"], "ppm": ref["ppm"], "intensity": ref["intensity"], "smiles": s})
    return refs

def quantify_mixture(ppm: List[float], intensity: List[float], selected_refs: List[Dict[str, Any]]) -> Dict[str, Any]:
    # ASICS expects 1 mixture and an array of reference spectra (each with ppm/intensity + name)
    res = asics_quantify(
        crude_ppm=ppm,
        crude_intensity=intensity,
        refs=[{"name": r["name"], "ppm": r["ppm"], "intensity": r["intensity"]} for r in selected_refs],
        nb_protons=None,            # optional: pass in later if available
        exclusion=None,
        max_shift=0.02,
        quant_method="FWER",
    )
    # res will contain concentrations + reconstructed spectra
    return res
