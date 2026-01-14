# app/tools_nmrbank.py
import os, glob, logging, re
from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem

# ----------------------------
# Logging (quiet by default)
# ----------------------------
_LOG_LEVEL = os.getenv("NMRBANK_LOG_LEVEL", "WARNING").upper()
logging.basicConfig(level=_LOG_LEVEL, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("tools_nmrbank")

# ----------------------------
# Globals (lazy)
# ----------------------------
_DB: Optional[pd.DataFrame] = None      # raw DataFrame
_LUT: Optional[Dict[str, dict]] = None  # canonical_smiles -> entry

# ----------------------------
# Path discovery
# ----------------------------
def _find_csv_path() -> Optional[str]:
    env = os.environ.get("NMRBANK_CSV")
    if env and os.path.exists(env):
        return env
    here = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = [
        os.path.join(here, "NMRBank", "NMRBank_data_with_SMILES_156621_in_225809.csv"),
        os.path.join(here, "NMRBank", "NMRBank", "NMRBank_data_with_SMILES_156621_in_225809.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    for p in glob.glob(os.path.join(here, "NMRBank", "**", "*with_SMILES*.csv"),
                       recursive=True):
        if os.path.exists(p):
            return p
    return None

# ----------------------------
# Helpers
# ----------------------------
_H1_RANGE: Tuple[float, float] = (0.0, 12.5)
_C13_RANGE: Tuple[float, float] = (0.0, 220.0)
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")

def _smiles_key(s: str) -> Optional[str]:
    try:
        mol = Chem.MolFromSmiles(str(s))
        return Chem.MolToSmiles(mol, canonical=True) if mol else None
    except Exception:
        return None

def _dedupe_and_sum(points: List[Tuple[float,float]], bin_width: float = 0.02):
    """
    Merge peaks that are within bin_width by summing intensities.
    Returns sorted lists (ppm_desc, inten).
    """
    if not points:
        return [], []
    # bin by rounding to two decimals (approx 0.01 ppm)
    bins: Dict[float, float] = {}
    for p, w in points:
        key = round(p, 2) if bin_width <= 0.02 else round(p / bin_width) * bin_width
        bins[key] = bins.get(key, 0.0) + float(w)
    ppm = sorted(bins.keys(), reverse=True)
    inten = [bins[p] for p in ppm]
    return ppm, inten

def _norm_to_max1(intensities: List[float]) -> List[float]:
    if not intensities:
        return intensities
    m = max(intensities)
    if m <= 0:
        return [1.0 for _ in intensities]
    return [v / m for v in intensities]

def _parse_h1_with_integrals(cell: str) -> Tuple[List[float], List[float]]:
    """
    Parse ¹H entries like:
      "7.26 (d, 2H, J=8.8 Hz); 3.78 (s, 3H)"
    Extract ppm inside 0..12.5 and use the nearest 'nH' token in the same
    clause as an approximate integral. Falls back to 1.0 if no 'nH'.
    """
    if not cell or pd.isna(cell):
        return [], []
    text = str(cell)

    points: List[Tuple[float, float]] = []
    # split clauses (commas, semicolons, slashes)
    for clause in re.split(r"[;/]", text):
        # further split by semicolon/comma boundaries but keep clause context
        for token in re.split(r"\s*;\s*|\s*,\s*(?=\d)", clause):
            nums = list(_NUM.finditer(token))
            if not nums:
                continue

            # pick first ppm-like number in range (skip obvious coupling constants)
            ppm_val = None
            for m in nums:
                val = float(m.group())
                # Look ahead/behind for "J" / "Hz" markers to avoid couplings
                head = token[max(0, m.start()-4): m.start()]
                tail = token[m.end(): m.end()+6]
                if "Hz" in tail or "J" in head or "J=" in head:
                    continue
                if _H1_RANGE[0] <= val <= _H1_RANGE[1]:
                    ppm_val = val
                    break
            if ppm_val is None:
                continue

            # try to find nH in the same token/segment
            mH = re.search(r"(\d+(?:\.\d+)?)\s*H\b", token, flags=re.IGNORECASE)
            weight = float(mH.group(1)) if mH else 1.0
            points.append((ppm_val, weight))

    ppm, inten = _dedupe_and_sum(points, bin_width=0.02)
    inten = _norm_to_max1(inten)  # normalize for plotting
    return ppm, inten

def _parse_h1_ppm(cell: str) -> List[float]:
    """
    Back-compat helper used by tests and legacy code: return only ¹H ppm values.
    Internally uses the same parsing as `_parse_h1_with_integrals` and
    discards the computed intensities.
    """
    ppm, _ = _parse_h1_with_integrals(cell)
    return ppm

def _parse_c13_ppm(cell: str) -> List[float]:
    """Parse ¹³C numbers in 0..220 ppm; no intensities available."""
    if not cell or pd.isna(cell):
        return []
    vals: List[float] = []
    for m in _NUM.finditer(str(cell)):
        v = float(m.group())
        if _C13_RANGE[0] <= v <= _C13_RANGE[1]:
            vals.append(v)
    return sorted(set(vals), reverse=True)

def _read_csv(csv_path: Optional[str]) -> pd.DataFrame:
    if not csv_path:
        log.warning("NMRBank CSV path not found. Set NMRBANK_CSV or place the CSV under NMRBank/.")
        return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        log.info("Loaded %d rows from CSV: %s", len(df), csv_path)
        return df
    except Exception as e:
        log.error("Error loading CSV: %s", e)
        return pd.DataFrame()

# ----------------------------
# Lazy loaders
# ----------------------------
def _load_db() -> pd.DataFrame:
    """
    Load the raw CSV once. Skip entirely when NMRBANK_SKIP_LOAD_FOR_TESTS=1.
    """
    global _DB
    if _DB is not None:
        return _DB

    if os.getenv("NMRBANK_SKIP_LOAD_FOR_TESTS") == "1":
        log.info("Skipping NMRBank CSV load for tests.")
        _DB = pd.DataFrame(columns=[
            "IUPAC Name", "SMILES", "Standardized SMILES", "Standardized_SMILES",
            "1H NMR chemical shifts", "13C NMR chemical shifts",
        ])
        return _DB

    csv_path = _find_csv_path()
    _DB = _read_csv(csv_path)
    return _DB
def _build_lut() -> Dict[str, dict]:
    """
    Build an in-memory LUT (canonical_smiles -> entry dict) from the NMRBank CSV.
    Raises a RuntimeError if the CSV cannot be loaded or is empty.
    """
    global _LUT
    if _LUT is not None:
        return _LUT

    df = _load_db()
    if df.empty:
        raise RuntimeError(
            "NMRBank CSV could not be loaded or is empty. "
            "Please set NMRBANK_CSV to the correct path."
        )

    lut: Dict[str, dict] = {}

    for _, row in df.iterrows():
        smi = (
            row.get("SMILES")
            or row.get("Standardized SMILES")
            or row.get("Standardized_SMILES")
        )
        if not smi or pd.isna(smi):
            continue

        key = _smiles_key(smi)
        if not key or key in lut:
            continue

        # Parse ¹H with approximate integrals (from "nH" tokens)
        h1_ppm, h1_int = _parse_h1_with_integrals(row.get("1H NMR chemical shifts", ""))
        c13_ppm = _parse_c13_ppm(row.get("13C NMR chemical shifts", ""))

        if not h1_ppm and not c13_ppm:
            continue

        name = row.get("IUPAC Name")
        if not name or pd.isna(name):
            name = key

        entry = {
            "name": str(name),
            "smiles": key,
        }
        if h1_ppm:
            entry["ppm_h1"] = h1_ppm
            entry["intensity_h1"] = h1_int if h1_int else [1.0] * len(h1_ppm)
        if c13_ppm:
            entry["ppm_c13"] = c13_ppm
            entry["intensity_c13"] = [1.0] * len(c13_ppm)

        # for backward-compat with code that expects 'ppm'/'intensity':
        if h1_ppm:
            entry["ppm"] = h1_ppm
            entry["intensity"] = entry["intensity_h1"]
        elif c13_ppm:
            entry["ppm"] = c13_ppm
            entry["intensity"] = entry["intensity_c13"]

        lut[key] = entry

    log.info("Built NMRBank LUT with %d unique compounds", len(lut))
    _LUT = lut
    return _LUT
     
# ----------------------------
# Public API
# ----------------------------
def get_reference_by_smiles(smiles: str) -> Optional[dict]:
    """
    Return a dict for the canonical SMILES with keys:
      - name, smiles
      - ppm_h1/intensity_h1 (if available)
      - ppm_c13/intensity_c13 (if available)
    For backward compatibility: also includes 'ppm'/'intensity' (¹H preferred).
    """
    key = _smiles_key(smiles)
    if not key:
        return None
    return _build_lut().get(key)

def warm_cache() -> int:
    """Force-load DB and LUT. Returns number of compounds in LUT."""
    return len(_build_lut())

def clear_cache() -> None:
    """Clear in-memory caches (tests/dev)."""
    global _DB, _LUT
    _DB = None
    _LUT = None
