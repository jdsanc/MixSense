"""
Wrapper tool to run Masserstein+Gurobi deconvolution using scripts/deconvolve_nmr.py

Usage from code:
    from .tools_deconvolve import deconvolve_spectra
    res = deconvolve_spectra(mix_ppm, mix_intensity, refs)

Where `refs` is a list of dicts with keys: name, ppm, intensity (and optionally protons).
Returns a dict with keys:
    - concentrations: { name: ratio }
    - raw: parsed JSON payload if available
    - stdout: full stdout text from the external script
"""
from __future__ import annotations

import os
import json
import tempfile
import subprocess
from typing import Dict, List, Any, Optional


def _write_xy_csv(path: str, x: List[float], y: List[float]) -> None:
    with open(path, "w") as f:
        for xi, yi in zip(x or [], y or []):
            f.write(f"{xi},{yi}\n")


def _parse_stdout(stdout: str) -> Dict[str, float]:
    """Parse either a JSON line (JSON: {...}) or the text table under
    'Estimated proportions:' and return a {name: value} mapping.
    """
    concentrations: Dict[str, float] = {}

    # Try JSON line first
    json_line = ""
    for ln in stdout.splitlines()[::-1]:
        if ln.startswith("JSON:"):
            json_line = ln[len("JSON:"):].strip()
            break
    if not json_line:
        start = stdout.rfind("{")
        end = stdout.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_line = stdout[start : end + 1]

    payload: Optional[Dict[str, Any]] = None
    if json_line:
        try:
            payload = json.loads(json_line)
            props = payload.get("proportions", {})
            for k, v in props.items():
                try:
                    concentrations[k] = float(v)
                except Exception:
                    pass
        except Exception:
            payload = None

    # Fallback to text table if needed
    if not concentrations:
        import re

        grab = False
        for ln in stdout.splitlines():
            if ln.strip().lower().startswith("estimated proportions"):
                grab = True
                continue
            if grab:
                m = re.match(r"\s*(.+?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", ln)
                if m:
                    try:
                        concentrations[m.group(1).strip()] = float(m.group(2))
                    except Exception:
                        pass

    return {k: v for k, v in concentrations.items()}


def deconvolve_spectra(
    mixture_ppm: List[float],
    mixture_intensity: List[float],
    refs: List[Dict[str, Any]],
    *,
    names: Optional[List[str]] = None,
    protons: Optional[List[int]] = None,
    threads: int = 8,
    quiet: bool = True,
) -> Dict[str, Any]:
    """Run Magnetstein deconvolution and return concentrations.

    refs: list of { name, ppm, intensity, [protons] }
    """
    # Prefer bundled app/tool_deconvolve_nmr.py, allow override via env
    script_path = os.environ.get("DECONVOLVE_SCRIPT")
    if not script_path:
        script_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "tool_deconvolve_nmr.py")
        )

    # Ensure names/protons lists
    n = len(refs or [])
    ref_names = names or [r.get("name", f"comp{i}") for i, r in enumerate(refs)]
    ref_protons = (
        protons
        if protons is not None and len(protons) == n
        else [int(r.get("protons", 1)) for r in refs]
    )

    with tempfile.TemporaryDirectory() as td:
        mix_path = os.path.join(td, "mixture.csv")
        _write_xy_csv(mix_path, mixture_ppm, mixture_intensity)

        comp_paths: List[str] = []
        for i, r in enumerate(refs):
            cpath = os.path.join(td, f"comp_{i}.csv")
            _write_xy_csv(cpath, r.get("ppm", []), r.get("intensity", []))
            comp_paths.append(cpath)

        cmd = [
            os.environ.get("PYTHON", "python"),
            script_path,
            mix_path,
            *comp_paths,
            "--protons",
            *[str(p) for p in ref_protons],
            "--names",
            *[str(nm) for nm in ref_names],
            "--threads",
            str(threads),
            "--json",
        ]
        if quiet:
            cmd += ["--quiet"]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout = (proc.stdout or "") + (proc.stderr or "")

        concentrations = _parse_stdout(stdout)
        raw = {}
        try:
            # If JSON line existed, it was parsed above; re-parse for return
            json_line = ""
            for ln in stdout.splitlines()[::-1]:
                if ln.startswith("JSON:"):
                    json_line = ln[len("JSON:"):].strip()
                    break
            if json_line:
                raw = json.loads(json_line)
        except Exception:
            raw = {}

    return {"concentrations": concentrations, "raw": raw, "stdout": stdout}


