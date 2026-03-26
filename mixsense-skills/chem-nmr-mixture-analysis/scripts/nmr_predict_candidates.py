#!/usr/bin/env python3
"""
Predict NMR reference candidates from a reaction description.

Workflow:
  1. Parse reactants/reagents from a reaction description (text file or string)
  2. Resolve chemical names -> canonical SMILES (RDKit direct parse, then PubChem REST)
  3. Predict reaction products via HuggingFace ReactionT5 API
  4. Look up 1H NMR reference spectra for all species (reactants + products) in NMRBank
  5. Save found spectra as CSV files ready for nmr_deconvolve.py / nmr_kinetics.py

Usage:
    # From a reaction description file
    python nmr_predict_candidates.py --reaction reaction.txt --nmrbank_csv /path/to/nmrbank.csv

    # Inline
    python nmr_predict_candidates.py \
        --reactants "camphor" \
        --reagents "NaBH4" \
        --nmrbank_csv /path/to/nmrbank.csv \
        --output_dir candidates/

Requirements:
    - rdkit, numpy, pandas, requests
    - HF_TOKEN env var (for ReactionT5 product prediction via HuggingFace API)
    - NMRBank CSV file (set via NMRBANK_CSV env var or --nmrbank_csv)
"""

import argparse
import json
import os
import pathlib
import re
import sys
import time

import numpy as np
import requests

def _smiles_from_rdkit(name: str):
    """Try to parse name directly as a SMILES string."""
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(name)
        if m:
            return Chem.MolToSmiles(m, canonical=True)
    except Exception:
        pass
    return None


def _smiles_from_pubchem(name: str) -> str | None:
    """Look up canonical SMILES via PubChem REST API (no token required)."""
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{requests.utils.quote(name)}/property/IsomericSMILES/JSON"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            smiles = data["PropertyTable"]["Properties"][0]["IsomericSMILES"]
            # Canonicalise with RDKit if available
            try:
                from rdkit import Chem
                m = Chem.MolFromSmiles(smiles)
                if m:
                    return Chem.MolToSmiles(m, canonical=True)
            except Exception:
                pass
            return smiles
    except Exception:
        pass
    return None


def resolve_name_to_smiles(name: str) -> str | None:
    """
    Convert a chemical name or SMILES string to canonical SMILES.
    Tries: (1) direct RDKit parse, (2) PubChem REST API.
    """
    can = _smiles_from_rdkit(name)
    if can:
        return can
    can = _smiles_from_pubchem(name)
    return can


_HF_API_URL = "https://router.huggingface.co/hf-inference/models/sagawa/ReactionT5v2-forward"


def _valid_smiles(s: str) -> bool:
    try:
        from rdkit import Chem
        return Chem.MolFromSmiles(s) is not None
    except Exception:
        return False


def predict_products(
    reactants_smiles: list[str],
    reagents_smiles: list[str],
    hf_token: str,
    n_best: int = 5,
) -> list[str]:
    """
    Call ReactionT5 via HuggingFace API to predict products.

    Args:
        reactants_smiles: List of reactant canonical SMILES.
        reagents_smiles: List of reagent canonical SMILES (catalysts, solvents etc).
        hf_token: HuggingFace API token.
        n_best: Number of top predictions to return.

    Returns:
        List of unique predicted product SMILES (deduplicated, multi-component split).
    """
    reactants_str = " . ".join(reactants_smiles)
    reagents_str = " . ".join(reagents_smiles) if reagents_smiles else ""
    prompt = f"{reactants_str} > {reagents_str} >"

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 64,
            "num_beams": max(n_best, 10),
            "num_return_sequences": n_best,
            "do_sample": False,
            "early_stopping": True,
        },
    }
    headers = {"Authorization": f"Bearer {hf_token}"}

    try:
        r = requests.post(_HF_API_URL, headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        response = r.json()
    except Exception as e:
        print(f"ERROR: ReactionT5 API call failed: {e}", file=sys.stderr)
        return []

    texts = []
    if isinstance(response, list):
        texts = [item.get("generated_text", "") for item in response]
    elif isinstance(response, dict) and "generated_text" in response:
        texts = [response["generated_text"]]

    # Extract unique component SMILES from all predictions
    seen = set()
    components = []
    for text in texts:
        prediction = text.split()[0].strip() if text.split() else ""
        for part in prediction.split("."):
            part = part.strip()
            if part and _valid_smiles(part) and part not in seen:
                seen.add(part)
                components.append(part)

    return components


_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
_H1_RANGE = (0.0, 12.5)

_lut_cache: dict | None = None


def _smiles_key(s: str) -> str | None:
    try:
        from rdkit import Chem
        m = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(m, canonical=True) if m else None
    except Exception:
        return None


def _parse_h1(cell: str):
    """Parse 1H NMR chemical shifts cell into (ppm_list, intensity_list)."""
    if not cell:
        return [], []
    try:
        import pandas as pd
        if pd.isna(cell):
            return [], []
    except Exception:
        pass

    points = []
    for clause in re.split(r"[;/]", str(cell)):
        for token in re.split(r"\s*;\s*|\s*,\s*(?=\d)", clause):
            nums = list(_NUM.finditer(token))
            if not nums:
                continue
            ppm_val = None
            for m in nums:
                val = float(m.group())
                head = token[max(0, m.start() - 4): m.start()]
                tail = token[m.end(): m.end() + 6]
                if "Hz" in tail or "J" in head or "J=" in head:
                    continue
                if re.match(r"\s*H\b", tail, flags=re.IGNORECASE):
                    continue
                if _H1_RANGE[0] <= val <= _H1_RANGE[1]:
                    ppm_val = val
                    break
            if ppm_val is None:
                continue
            mH = re.search(r"(\d+(?:\.\d+)?)\s*H\b", token, flags=re.IGNORECASE)
            weight = float(mH.group(1)) if mH else 1.0
            points.append((ppm_val, weight))

    if not points:
        return [], []

    bins: dict[float, float] = {}
    for p, w in points:
        key = round(p, 2)
        bins[key] = bins.get(key, 0.0) + w
    ppm = sorted(bins.keys(), reverse=True)
    inten = [bins[p] for p in ppm]
    mx = max(inten)
    inten = [v / mx for v in inten] if mx > 0 else inten
    return ppm, inten


def _load_nmrbank_lut(csv_path: str) -> dict:
    global _lut_cache
    if _lut_cache is not None:
        return _lut_cache

    import pandas as pd
    print(f"Loading NMRBank from {csv_path} ...", flush=True)
    df = pd.read_csv(csv_path, low_memory=False)

    lut = {}
    for _, row in df.iterrows():
        smi = row.get("SMILES") or row.get("Standardized SMILES") or row.get("Standardized_SMILES")
        if not smi or (isinstance(smi, float) and np.isnan(smi)):
            continue
        key = _smiles_key(str(smi))
        if not key or key in lut:
            continue
        ppm, inten = _parse_h1(row.get("1H NMR chemical shifts", ""))
        if not ppm:
            continue
        name = row.get("IUPAC Name")
        if not name or (isinstance(name, float) and np.isnan(name)):
            name = key
        lut[key] = {"name": str(name), "smiles": key, "ppm": ppm, "intensity": inten}

    print(f"NMRBank loaded: {len(lut)} compounds with 1H spectra.", flush=True)
    _lut_cache = lut
    return lut


def lookup_nmr_reference(smiles: str, csv_path: str) -> dict | None:
    """Look up 1H NMR reference for a canonical SMILES in NMRBank."""
    lut = _load_nmrbank_lut(csv_path)
    key = _smiles_key(smiles)
    if not key:
        return None
    return lut.get(key)


def parse_reaction_text(text: str) -> tuple[list[str], list[str]]:
    """
    Very simple parser for plain-English reaction descriptions.

    Returns (reactant_names, reagent_names).

    Recognises patterns like:
      "Reduction of camphor with NaBH4"
      "Reactants: camphor; Reagents: NaBH4, THF"
      "camphor + NaBH4"
    If no structure is found, treats all comma/+/semicolon-separated tokens as reactants.
    """
    text = text.strip()

    # Try structured format first: "Reactants: ...; Reagents: ..."
    reactants, reagents = [], []
    m_react = re.search(r"[Rr]eactants?\s*[:\-]\s*([^;\n]+)", text)
    m_reag = re.search(r"[Rr]eagents?\s*[:\-]\s*([^;\n]+)", text)
    if m_react:
        reactants = [t.strip() for t in re.split(r"[,;+]", m_react.group(1)) if t.strip()]
    if m_reag:
        reagents = [t.strip() for t in re.split(r"[,;+]", m_reag.group(1)) if t.strip()]
    if reactants:
        return reactants, reagents

    # Try "X with Y" / "X using Y"
    m_with = re.search(r"(?:of|reduction of|oxidation of|reaction of)\s+(.+?)\s+(?:with|using)\s+(.+)", text, re.IGNORECASE)
    if m_with:
        reactants = [t.strip() for t in re.split(r"[,;+]", m_with.group(1)) if t.strip()]
        reagents = [t.strip() for t in re.split(r"[,;+]", m_with.group(2)) if t.strip()]
        return reactants, reagents

    # Fallback: treat everything as reactants
    all_tokens = [t.strip() for t in re.split(r"[,;+]", text) if t.strip()]
    return all_tokens, []


def main():
    ap = argparse.ArgumentParser(
        description="Predict NMR reference candidates from a reaction description."
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--reaction", metavar="FILE",
                     help="Plain-text reaction description file (e.g. reaction.txt)")
    src.add_argument("--reactants", metavar="NAMES",
                     help="Comma-separated reactant names or SMILES")
    ap.add_argument("--reagents", metavar="NAMES", default="",
                    help="Comma-separated reagent/catalyst names or SMILES")
    ap.add_argument("--nmrbank_csv", metavar="FILE",
                    default=os.environ.get("NMRBANK_CSV", ""),
                    help="Path to NMRBank CSV (or set NMRBANK_CSV env var)")
    ap.add_argument("--output_dir", default="candidates",
                    help="Directory for output reference CSVs and manifest (default: candidates/)")
    ap.add_argument("--n_best", type=int, default=5,
                    help="Number of ReactionT5 product predictions (default: 5)")
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN", ""),
                    help="HuggingFace API token (or set HF_TOKEN env var)")
    ap.add_argument("--skip_prediction", action="store_true",
                    help="Skip product prediction; only look up spectra for reactants")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    # --- Parse input ---
    if args.reaction:
        text = pathlib.Path(args.reaction).read_text()
        reactant_names, reagent_names = parse_reaction_text(text)
    elif args.reactants:
        reactant_names = [t.strip() for t in args.reactants.split(",") if t.strip()]
        reagent_names = [t.strip() for t in args.reagents.split(",") if t.strip()] if args.reagents else []
    else:
        ap.error("Provide --reaction FILE or --reactants NAMES")

    if not args.quiet:
        print(f"Reactants: {reactant_names}")
        print(f"Reagents:  {reagent_names}")

    # --- Resolve names -> SMILES ---
    if not args.quiet:
        print("\nResolving names to SMILES...")
    reactant_smiles, reagent_smiles = [], []
    name_to_smiles = {}

    for name in reactant_names:
        smi = resolve_name_to_smiles(name)
        if smi:
            reactant_smiles.append(smi)
            name_to_smiles[smi] = name
            if not args.quiet:
                print(f"  {name} -> {smi}")
        else:
            print(f"  WARNING: Could not resolve '{name}' to SMILES", file=sys.stderr)

    for name in reagent_names:
        smi = resolve_name_to_smiles(name)
        if smi:
            reagent_smiles.append(smi)
            name_to_smiles[smi] = name
            if not args.quiet:
                print(f"  {name} -> {smi} (reagent)")
        else:
            print(f"  WARNING: Could not resolve reagent '{name}' to SMILES", file=sys.stderr)

    if not reactant_smiles:
        print("ERROR: No reactants could be resolved to SMILES. Exiting.", file=sys.stderr)
        sys.exit(1)

    # --- Predict products ---
    predicted_smiles = []
    if not args.skip_prediction:
        if not args.hf_token:
            print("WARNING: No HF_TOKEN set. Skipping product prediction.", file=sys.stderr)
            print("         Set HF_TOKEN or use --skip_prediction to suppress this.", file=sys.stderr)
        else:
            if not args.quiet:
                print(f"\nPredicting products (ReactionT5, n_best={args.n_best})...")
            predicted_smiles = predict_products(
                reactant_smiles, reagent_smiles, args.hf_token, n_best=args.n_best
            )
            if not args.quiet:
                print(f"  {len(predicted_smiles)} unique product components predicted")
                for s in predicted_smiles:
                    print(f"    {s}")

    # --- All species to look up ---
    all_smiles = list(dict.fromkeys(reactant_smiles + predicted_smiles))  # preserve order, dedupe

    # --- NMRBank lookup ---
    if not args.nmrbank_csv:
        print("ERROR: No NMRBank CSV specified. Use --nmrbank_csv or set NMRBANK_CSV env var.", file=sys.stderr)
        sys.exit(1)
    if not pathlib.Path(args.nmrbank_csv).exists():
        print(f"ERROR: NMRBank CSV not found: {args.nmrbank_csv}", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print(f"\nLooking up NMR references for {len(all_smiles)} species...")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    found = []
    missing = []

    for smiles in all_smiles:
        ref = lookup_nmr_reference(smiles, args.nmrbank_csv)
        if ref:
            name = name_to_smiles.get(smiles, ref["name"])
            safe_name = re.sub(r"[^\w\-]", "_", name)[:40]
            csv_path = out_dir / f"{safe_name}.csv"
            ppm = ref["ppm"]
            inten = ref["intensity"]
            arr = np.column_stack([ppm, inten])
            np.savetxt(csv_path, arr, delimiter=",", fmt="%.6f")
            found.append({
                "smiles": smiles,
                "name": name,
                "nmrbank_name": ref["name"],
                "n_peaks": len(ppm),
                "csv": str(csv_path),
            })
            if not args.quiet:
                print(f"  FOUND  {name:30s}  ({len(ppm)} peaks) -> {csv_path}")
        else:
            label = name_to_smiles.get(smiles, smiles)
            missing.append({"smiles": smiles, "name": label})
            if not args.quiet:
                print(f"  MISS   {label} ({smiles})")

    # --- Save manifest ---
    manifest = {
        "reactants": [{"name": n, "smiles": s} for n, s in zip(reactant_names, reactant_smiles)],
        "reagents": [{"name": n, "smiles": s} for n, s in zip(reagent_names, reagent_smiles)],
        "predicted_products": predicted_smiles,
        "found_references": found,
        "missing_references": missing,
    }
    manifest_path = out_dir / "candidates.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # --- Summary ---
    print(f"\nDone: {len(found)} references found, {len(missing)} missing.")
    print(f"Reference CSVs and manifest -> {out_dir}/")

    if found:
        print("\nReady for deconvolution:")
        ref_args = " ".join(f'"{e["csv"]}"' for e in found)
        name_args = " ".join(f'"{e["name"]}"' for e in found)
        print(f"  uv run python mixsense-skills/chem-nmr-mixture-analysis/scripts/nmr_deconvolve.py \\")
        print(f"    crude.csv {ref_args} \\")
        print(f"    --names {name_args} \\")
        print(f"    --baseline-correct --json")

    if missing:
        print(f"\nNo NMRBank entry for: {[e['name'] for e in missing]}")
        print("  -> Provide measured or digitized reference CSVs for these manually.")


if __name__ == "__main__":
    main()
