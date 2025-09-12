import os, zipfile, json, io
import pandas as pd
import re
from typing import Dict, List, Optional, Tuple
from rdkit import Chem

# Use the CSV file directly instead of JSON zip
_CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "NMRBank",
    "NMRBank",
    "NMRBank_data_with_SMILES_156621_in_225809.csv",
)


def _smiles_key(s: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None


def _parse_nmr_shifts(shift_text: str) -> List[float]:
    """Parse NMR chemical shifts from text format to list of floats."""
    if not shift_text or pd.isna(shift_text) or shift_text == "N/A":
        return []

    # Extract numbers from the text, handling various formats
    # Look for patterns like "1.55 (d, 3H, J = 6.7 Hz)" or "28.34, 30.18, 34.27"
    numbers = re.findall(r"-?\d+\.?\d*", str(shift_text))
    try:
        return [
            float(num) for num in numbers if float(num) >= -50 and float(num) <= 300
        ]  # Reasonable NMR range
    except (ValueError, TypeError):
        return []


def _read_csv_data(csv_path: str) -> pd.DataFrame:
    """Read and return the NMRBank CSV data."""
    print(f"Loading CSV data from: {csv_path}")
    try:
        # Read CSV with proper handling of large file
        df = pd.read_csv(csv_path, low_memory=False)
        print(f"Loaded {len(df)} rows from CSV")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return pd.DataFrame()


_CACHE = None


def _load_all() -> Dict[str, dict]:
    global _CACHE
    if _CACHE is None:
        df = _read_csv_data(_CSV_PATH)
        lut = {}

        for _, row in df.iterrows():
            # Try to get SMILES from either column
            smi = row.get("SMILES") or row.get("Standardized SMILES")
            if pd.isna(smi) or not smi:
                continue

            key = _smiles_key(smi)
            if not key or key in lut:
                continue

            # Parse NMR data
            h1_shifts = _parse_nmr_shifts(row.get("1H NMR chemical shifts", ""))
            c13_shifts = _parse_nmr_shifts(row.get("13C NMR chemical shifts", ""))

            # Only include entries with valid NMR data
            if h1_shifts or c13_shifts:
                lut[key] = {
                    "name": row.get("IUPAC Name", key),
                    "smiles": key,
                    "ppm": h1_shifts + c13_shifts,  # Combine both 1H and 13C shifts
                    "intensity": [1.0]
                    * len(h1_shifts + c13_shifts),  # Default intensity
                }

        print(f"Loaded {len(lut)} unique compounds with NMR data")
        _CACHE = lut
    return _CACHE


def get_reference_by_smiles(smiles: str) -> Optional[dict]:
    key = _smiles_key(smiles)
    if not key:
        return None
    return _load_all().get(key)
