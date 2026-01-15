"""
ReactionT5v2 forward prediction module.
Predicts products from reactants using the sagawa/ReactionT5v2-forward model.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from rdkit import Chem

MODEL_NAME = "sagawa/ReactionT5v2-forward"

# Lazy loading to avoid loading model if not used
_tok = None
_mdl = None


def _load_model():
    global _tok, _mdl
    if _tok is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()
    return _tok, _mdl


def valid_smiles(s: str) -> bool:
    try:
        return Chem.MolFromSmiles(s) is not None
    except Exception:
        return False


def predict_products(
    reactants: str,
    reagents: str = "",
    beams: int = 10,
    n_best: int = 5,
    max_new_tokens: int = 64,
):
    """
    Return up to n_best product SMILES from ReactionT5v2.

    Args:
        reactants: Dot-separated reactant SMILES (e.g., "COc1ccccc1.BrBr")
        reagents: Optional reagents (e.g., "Br[Fe](Br)Br" for FeBr3)
        beams: Beam search width
        n_best: Number of predictions to return
        max_new_tokens: Max tokens in output

    Returns:
        List of valid product SMILES strings
    """
    tok, mdl = _load_model()

    # Model expects: "REACTANT:<reactants> REAGENT:<reagents>"
    prompt = f"REACTANT:{reactants} REAGENT:{reagents if reagents else ' '}"
    inp = tok(prompt, return_tensors="pt")

    with torch.no_grad():
        out = mdl.generate(
            **inp,
            num_beams=beams,
            num_return_sequences=n_best,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
            length_penalty=0.8
        )

    preds = [tok.decode(o, skip_special_tokens=True).replace(" ", "").rstrip(".") for o in out]
    preds = [p for p in preds if valid_smiles(p)]
    preds = list(dict.fromkeys(preds))
    return preds


# Pattern for ester detection
PAT_ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2][CX4]")


def _count_matches(mols, patt):
    return sum(len(m.GetSubstructMatches(patt)) for m in mols)


def add_water_for_new_esters(reactants_smi: str, product_smi: str) -> str:
    """
    Appends one '.O' (water) per newly formed ester linkage.
    """
    r_mols = [Chem.MolFromSmiles(s) for s in reactants_smi.split(".") if s]
    p_mols = [Chem.MolFromSmiles(s) for s in product_smi.split(".") if s]
    if any(m is None for m in r_mols + p_mols):
        return product_smi

    formed = max(0, _count_matches(p_mols, PAT_ESTER) - _count_matches(r_mols, PAT_ESTER))
    if formed > 0 and ".O" not in ("." + product_smi + "."):
        return product_smi + ".O" * formed
    return product_smi


def unique_components_from_balanced(balanced_preds):
    """
    Extract unique components from balanced prediction strings.
    """
    seen, out = set(), []
    for s in balanced_preds:
        for part in s.split('.'):
            p = part.strip()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
    return out


if __name__ == "__main__":
    # Example: acetic acid + ethanol (Fischer esterification)
    reactants = "CC(=O)O.CCO"
    reagents = ""

    preds = predict_products(reactants, reagents, beams=10, n_best=5)
    balanced_preds = [add_water_for_new_esters(reactants, p) for p in preds]
    balanced_preds = list(dict.fromkeys(balanced_preds))

    print("Raw model predictions (top-k):")
    for i, p in enumerate(preds, 1):
        print(f"{i}. {p}")

    print("\nBalanced predictions (H2O added when ester formed):")
    for i, p in enumerate(balanced_preds, 1):
        print(f"{i}. {p}")

    all_unique = unique_components_from_balanced(balanced_preds)
    print(f'\nUnique potential components: {all_unique}')
