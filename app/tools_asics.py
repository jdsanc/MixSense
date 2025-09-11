from __future__ import annotations
import os, json, tempfile, subprocess
from typing import List, Dict, Tuple, Any
import numpy as np, pandas as pd

R_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "r", "asics_quantify.R")

def _write_csv(ppm: List[float], cols: Dict[str, List[float]], path: str):
    df = pd.DataFrame({"ppm": ppm})
    for name, y in cols.items():
        df[name] = y
    df.to_csv(path, index=False)

def asics_quantify(
    crude_ppm: List[float],
    crude_intensity: List[float],
    refs: List[Dict[str, Any]],  # each: {"name": str, "ppm": [...], "intensity": [...]}
    nb_protons: Dict[str, int] | None = None,
    exclusion_ranges: List[Tuple[float,float]] | None = None,
    max_shift: float = 0.02,
    quant_method: str = "FWER",
    r_script: str = R_SCRIPT,
) -> Dict[str, Any]:
    ppm = np.array(crude_ppm)
    # resample refs to crude grid
    ref_cols = {}
    order = []
    for r in refs:
        y = np.interp(ppm, np.array(r["ppm"]), np.array(r["intensity"]), left=0.0, right=0.0)
        ref_cols[r["name"]] = y.tolist()
        order.append(r["name"])

    with tempfile.TemporaryDirectory() as td:
        mix_csv = os.path.join(td, "mixture.csv")
        lib_csv = os.path.join(td, "library.csv")
        _write_csv(crude_ppm, {"mixture": crude_intensity}, mix_csv)
        _write_csv(crude_ppm, ref_cols, lib_csv)

        nbp_json  = json.dumps([int(nb_protons.get(n, 1)) for n in order]) if nb_protons else json.dumps([])
        excl_json = json.dumps(exclusion_ranges or [[4.5, 5.1]])

        cmd = [
            "Rscript", r_script,
            "--mixture_csv", mix_csv,
            "--library_csv", lib_csv,
            "--nb_protons_json", nbp_json,
            "--exclusion_json", excl_json,
            "--max_shift", str(max_shift),
            "--quant_method", quant_method,
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        res = json.loads(out)

    # map first (and usually only) mixture row to list of components
    vals = res["quantification"][0] if isinstance(res["quantification"], list) else list(res["quantification"].values())[0]
    comps = [{"name": n, "fraction": float(v)} for n, v in zip(res["colnames"], vals)]
    return {"components": comps, "raw": res}
