#Need Transformers installed to run
#Not set up yet to run with agent. This should just take in the reactants pulled from the agent


# !pip install -U transformers torch rdkit-pypi

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from rdkit import Chem

# =========================
# Model: ReactionT5v2 (forward)
# =========================
MODEL_NAME = "sagawa/ReactionT5v2-forward"  
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()

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
    Input format required by this checkpoint:
        'REACTANT:<dot-separated reactants> REAGENT:<dot-separated reagents or single space>'
    """
    prompt = f"REACTANT:{reactants} REAGENT:{reagents if reagents else ' '}"
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = mdl.generate(
            **inp,
            num_beams=beams,
            num_return_sequences=n_best,
            max_new_tokens=max_new_tokens,
            early_stopping=True,   # avoid run-ons
            length_penalty=0.8     # discourage overly long strings
        )
    preds = [tok.decode(o, skip_special_tokens=True).replace(" ", "").rstrip(".") for o in out]
    # Basic cleanup: validity + de-dup (preserve order)
    preds = [p for p in preds if valid_smiles(p)]
    preds = list(dict.fromkeys(preds))
    return preds

# =========================
# Tiny post-processor: add H2O for each new ester bond formed
# (works even if the prediction includes leftover starting material)
# =========================
PAT_ESTER = Chem.MolFromSmarts("[CX3](=O)[OX2][CX4]")   # R–C(=O)–O–R'

def _count_matches(mols, patt):
    return sum(len(m.GetSubstructMatches(patt)) for m in mols)

def add_water_for_new_esters(reactants_smi: str, product_smi: str) -> str:
    """
    Appends one '.O' (water) per newly formed ester linkage.
    - Counts ester substructures in reactants vs. products.
    - If more in products, adds that many waters (unless already present).
    """
    r_mols = [Chem.MolFromSmiles(s) for s in reactants_smi.split(".") if s]
    p_mols = [Chem.MolFromSmiles(s) for s in product_smi.split(".") if s]
    if any(m is None for m in r_mols + p_mols):
        return product_smi  # fallback: don't modify

    formed = max(0, _count_matches(p_mols, PAT_ESTER) - _count_matches(r_mols, PAT_ESTER))
    if formed > 0 and ".O" not in ("." + product_smi + "."):
        return product_smi + ".O" * formed
    return product_smi

# =========================
# Example: acetic acid + ethanol (Fischer esterification)
# =========================
reactants = "CC(=O)O.CCO"     # acetic acid + ethanol (example we are using test workflow) 
reagents  = ""                # optional: "OS(=O)(=O)O" for H2SO4 catalyst

preds = predict_products(reactants, reagents, beams=10, n_best=5)
balanced_preds = [add_water_for_new_esters(reactants, p) for p in preds]
# (optional) de-dup again after balancing
balanced_preds = list(dict.fromkeys(balanced_preds))

print("Raw model predictions (top-k):")
for i, p in enumerate(preds, 1):
    print(f"{i}. {p}")

print("\nBalanced predictions (H2O added when ester formed):")
for i, p in enumerate(balanced_preds, 1):
    print(f"{i}. {p}")

def unique_components_from_balanced(balanced_preds):
    """
    balanced_preds: list of dot-separated product strings (already de-duped across predictions)
      e.g., ['CCOC(=O)C.O', 'CC(=O)O.CCOC(C)=O.O']
    returns: order-preserving unique components
      e.g., ['CCOC(=O)C', 'O', 'CC(=O)O', 'CCOC(C)=O']
    """
    seen, out = set(), []
    for s in balanced_preds:
        for part in s.split('.'):
            p = part.strip()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
    return out


# usage
all_unique_components = unique_components_from_balanced(balanced_preds)

print(f'Unique potential components: {all_unique_components}')
