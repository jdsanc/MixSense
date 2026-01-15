# app/tools_nmr2structure.py
"""
Wrapper to call rxn4chemistry/nmr-to-structure inference via OpenNMT.
You must supply:
  - a trained model checkpoint (.pt)
  - a tokenized `src` file in their expected text format.

For hackathon demo, you can point to their tiny example model/src.
"""
from typing import List
import subprocess, tempfile, os, textwrap

def predict_structures_from_token_lines(
    token_lines: List[str],      # one tokenized spectrum per line (their 'src' format)
    model_path: str,             # path/to/model_step_*.pt
    n_best: int = 5,
    beam_size: int = 10,
    gpu: int = -1,               # -1 CPU, or 0 for GPU
) -> List[List[str]]:
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as src_f:
        src_f.write("\n".join(token_lines))
        src_path = src_f.name

    with tempfile.NamedTemporaryFile(mode="r", delete=False) as out_f:
        out_path = out_f.name

    cmd = [
        "onmt_translate",
        "-model", model_path,
        "-src", src_path,
        "-output", out_path,
        "-beam_size", str(beam_size),
        "-n_best", str(n_best),
    ]
    if gpu >= 0:
        cmd += ["-gpu", str(gpu)]
    subprocess.run(cmd, check=True)

    # Output: n_best lines per input; parse first SMILES from each line
    preds_by_row = []
    with open(out_path, "r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    for i in range(0, len(lines), n_best):
        group = lines[i:i+n_best]
        smiles = [g.split()[0] for g in group]
        preds_by_row.append(smiles)
    os.unlink(src_path); os.unlink(out_path)
    return preds_by_row
