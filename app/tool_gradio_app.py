# run via python -m app.gradio_app
# ---------------------------------------
# Gradio UI for: proposal -> reference selection -> quantification (+ time-series)
# Backends: ASICS (existing) or Magnetstein (robust OT)
# Optional: NMR->Structure (rxn4chemistry/nmr-to-structure via OpenNMT)
# ---------------------------------------
import os, json, re
import numpy as np
import pandas as pd
import gradio as gr

from typing import List, Dict, Any, Tuple

# --- our app modules ---
from .agent import normalize_smiles_list, step_propose, load_refs_for_species
from .tools_reactiont5 import get_unique_components
from .tools_asics import asics_quantify

try:
    from .tools_magnetstein import (
        quantify_single as magnet_quant_single,
        quantify_timeseries as magnet_quant_series,
    )

    _HAS_MAGNETSTEIN = True
except Exception:
    _HAS_MAGNETSTEIN = False

try:
    from .tools_nmr2structure import predict_structures_from_token_lines

    _HAS_NMR2STRUCT = True
except Exception:
    _HAS_NMR2STRUCT = False

try:
    from .tools_nmrdb import get_reference_by_smiles_nmrdb

    _HAS_NMRDB = True
except Exception:
    _HAS_NMRDB = False


# ----------------------------
# Utilities
# ----------------------------
def parse_csv(file) -> Dict[str, List[float]]:
    """expects CSV with two columns: ppm,intensity"""
    if file is None:
        return {"ppm": [], "intensity": []}
    df = pd.read_csv(file.name if hasattr(file, "name") else file)
    df = df.rename(columns={df.columns[0]: "ppm", df.columns[1]: "intensity"})
    return {
        "ppm": df["ppm"].astype(float).tolist(),
        "intensity": df["intensity"].astype(float).tolist(),
    }


def parse_timeseries(files: List) -> Tuple[List[float], List[Dict[str, List[float]]]]:
    """multiple CSVs; we extract time as the first number in the filename"""
    items = []
    num_re = re.compile(r"(\d+(?:\.\d+)?)")
    for f in files or []:
        name = getattr(f, "name", "t0.csv")
        m = num_re.search(os.path.basename(name))
        t = float(m.group(1)) if m else 0.0
        items.append((t, parse_csv(f)))
    items.sort(key=lambda x: x[0])
    times = [t for t, _ in items]
    mixes = [m for _, m in items]
    return times, mixes


# ----------------------------
# Single-sample pipeline
# ----------------------------
def run_proposal(reactants_text, reagents_text, topk):
    rxs = normalize_smiles_list(reactants_text)
    if not rxs:
        return rxs, [], "No valid reactant SMILES found.", pd.DataFrame()
    products = step_propose(rxs, reagents_text, topk=topk)

    # Get unique components from all predictions (enhanced functionality from reactants_to_products)
    unique_components = get_unique_components(products)
    species = rxs + unique_components  # Use unique components instead of raw products

    refs = load_refs_for_species(species)
    msg = (
        f"Reactants: {rxs}\n"
        f"Proposed products: {products}\n"
        f"Unique components: {unique_components}\n"
        f"Found refs: {[r['name'] for r in refs]}"
    )
    table = pd.DataFrame(
        {
            "species_smiles": species,
            "in_library": [s in [r["smiles"] for r in refs] for s in species],
        }
    )
    return rxs, products, msg, table


def run_quantify(ppm_csv, reactants_list, products_list, selected_rows, backend_choice):
    mix = parse_csv(ppm_csv)
    if not mix["ppm"]:
        return "No mixture CSV provided.", None, None

    species = (reactants_list or []) + (products_list or [])
    if not species:
        return "No species available. Run 'Propose products' first.", None, None

    # Subset by selected rows if any:
    if selected_rows:
        species = [species[i] for i in selected_rows if 0 <= i < len(species)]

    refs = load_refs_for_species(species)
    if not refs:
        return "No reference spectra found for selected species.", None, None

    if backend_choice == "Magnetstein":
        if not _HAS_MAGNETSTEIN:
            return "Magnetstein not available (install or add submodule).", None, None
        res = magnet_quant_single(mix["ppm"], mix["intensity"], refs)
    else:
        res = asics_quantify(
            crude_ppm=mix["ppm"],
            crude_intensity=mix["intensity"],
            refs=[
                {"name": r["name"], "ppm": r["ppm"], "intensity": r["intensity"]}
                for r in refs
            ],
            nb_protons=None,
            exclusion_ranges=None,
            max_shift=0.02,
            quant_method="FWER",
        )

    # Handle different return formats from ASICS vs Magnetstein
    if backend_choice == "ASICS":
        # ASICS returns {"components": [{"name": str, "fraction": float}], "raw": {...}}
        conc = {comp["name"]: comp["fraction"] for comp in res.get("components", [])}
        reconstructed_intensity = [0] * len(
            mix["ppm"]
        )  # ASICS doesn't provide reconstruction
    else:
        # Magnetstein returns {"concentrations": {...}, "reconstructed": {...}}
        conc = res.get("concentrations", {})
        reconstructed_intensity = res.get("reconstructed", {}).get(
            "intensity", [0] * len(mix["ppm"])
        )

    df = pd.DataFrame(
        {"species": list(conc.keys()), "relative_conc": list(conc.values())}
    ).sort_values("relative_conc", ascending=False)

    overlay = {
        "ppm": mix["ppm"],
        "mixture": mix["intensity"],
        "reconstructed": reconstructed_intensity,
    }
    return f"OK ({backend_choice})", df, json.dumps(overlay)


# ----------------------------
# Optional: NMR -> Structure
# ----------------------------
def run_nmr2struct(toggle, model_file, src_file, n_best, beam_size, gpu_idx):
    if not toggle:
        return "Disabled.", []
    if not _HAS_NMR2STRUCT:
        return "NMR->Structure not available (install OpenNMT & tool).", []
    if model_file is None or src_file is None:
        return "Provide both a model .pt and a tokenized src .txt.", []

    with open(src_file.name, "r") as fh:
        token_lines = [ln.strip() for ln in fh if ln.strip()]
    preds = predict_structures_from_token_lines(
        token_lines=token_lines,
        model_path=model_file.name,
        n_best=int(n_best),
        beam_size=int(beam_size),
        gpu=int(gpu_idx),
    )
    # Flatten first item only (demo): show top-n for first line
    return "Done.", json.dumps(preds[0] if preds else [])


# ----------------------------
# Time-series (Magnetstein)
# ----------------------------
def run_timeseries(species_text, files):
    if not _HAS_MAGNETSTEIN:
        return "Magnetstein not available.", None, None
    species = normalize_smiles_list(species_text)
    if not species:
        return "No species provided.", None, None
    refs = load_refs_for_species(species)
    if not refs:
        return "No references found for the given species.", None, None

    times, mixes = parse_timeseries(files)
    if not times:
        return "No timeseries CSVs uploaded.", None, None

    res = magnet_quant_series(times, mixes, refs)
    # Build a tidy DF for plotting
    prop = res["proportions"]  # dict: name -> series
    df = pd.DataFrame({"time": res["times"]})
    for k, v in prop.items():
        df[k] = v
    # Return a CSV preview and a JSON payload
    return "OK (timeseries)", df, df.to_json(orient="records")


# ----------------------------
# Gradio layout
# ----------------------------
with gr.Blocks(
    title="Rxn → Products → NMR Quant (ASICS/Magnetstein) + NMR→Structure"
) as demo:
    gr.Markdown(
        "### Reaction proposal → Reference selection → Quantification\nSelect a backend and optionally run NMR→Structure."
    )

    with gr.Tabs():
        with gr.Tab("Single sample"):
            # --- Inputs row
            with gr.Row():
                reactants = gr.Textbox(
                    label="Reactants (SMILES or names, comma-separated)",
                    placeholder="anisole, Br2",
                )
                reagents = gr.Textbox(
                    label="Reagents/conditions (optional)", placeholder="FeBr3"
                )
                topk = gr.Slider(1, 10, value=5, step=1, label="Top-k products")

            # --- Backend dropdown (NEW) ---
            backend = gr.Dropdown(
                choices=["ASICS", "Magnetstein"],
                value="ASICS",
                label="Quantification backend",
            )

            propose_btn = gr.Button("Propose products")
            rxs_state = gr.State([])
            prods_state = gr.State([])

            table = gr.Dataframe(interactive=False)
            msg = gr.Textbox(label="Log", interactive=False, lines=4)

            # Mixture CSV + selection and quantify
            with gr.Row():
                ppm_csv = gr.File(
                    label="Upload mixture CSV (two columns: ppm,intensity)"
                )
                selected_rows = gr.Textbox(
                    label="Row indices to include (comma-separated; leave empty for all)",
                    value="",
                )
            quantify_btn = gr.Button("Quantify")

            conc_table = gr.Dataframe(
                label="Estimated concentrations", interactive=False
            )
            overlay_json = gr.Textbox(visible=False)

            # --- Optional NMR->Structure (NEW) ---
            gr.Markdown("#### Optional: NMR → Structure (rxn4chemistry)")
            nmr2s_toggle = gr.Checkbox(label="Enable NMR→Structure", value=False)
            with gr.Row(visible=True):
                model_pt = gr.File(label="OpenNMT checkpoint (.pt)")
                src_txt = gr.File(label="Tokenized src.txt (one spectrum per line)")
                n_best = gr.Slider(1, 10, value=5, step=1, label="n_best")
                beam_sz = gr.Slider(1, 20, value=10, step=1, label="beam_size")
                gpu_idx = gr.Number(value=-1, label="GPU index (-1=CPU)", precision=0)
            nmr2s_btn = gr.Button("Run NMR→Structure")
            nmr2s_log = gr.Textbox(label="NMR→Structure status", interactive=False)
            nmr2s_list = gr.Textbox(label="Top predictions (JSON)", interactive=False)

        with gr.Tab("Time-series (Magnetstein)"):
            gr.Markdown(
                "Upload multiple CSVs named with times, e.g. `t0.csv`, `t5.csv`, `10.0min.csv`."
            )
            series_species = gr.Textbox(
                label="Species to fit (SMILES or names, comma-separated)",
                placeholder="anisole, Brc1ccc(OC)cc1 (p-bromoanisole), Brc1cc(OC)ccc1 (o-bromoanisole)",
            )
            series_files = gr.File(
                file_count="multiple", label="Upload multiple CSVs (ppm,intensity)"
            )
            series_btn = gr.Button("Quantify time-series (Magnetstein)")
            series_log = gr.Textbox(label="Status", interactive=False)
            series_df = gr.Dataframe(
                label="Proportions vs time (preview)", interactive=False
            )
            series_json = gr.Textbox(visible=False)

    # --- Wire events ---
    propose_btn.click(
        run_proposal,
        inputs=[reactants, reagents, topk],
        outputs=[rxs_state, prods_state, msg, table],
    )

    def _parse_indices(text):
        if not text.strip():
            return None
        try:
            return [int(x.strip()) for x in text.split(",") if x.strip().isdigit()]
        except Exception:
            return None

    quantify_btn.click(
        lambda csv, rxs, prods, idx_text, backend_choice: run_quantify(
            csv, rxs, prods, _parse_indices(idx_text), backend_choice
        ),
        inputs=[ppm_csv, rxs_state, prods_state, selected_rows, backend],
        outputs=[msg, conc_table, overlay_json],
    )

    nmr2s_btn.click(
        run_nmr2struct,
        inputs=[nmr2s_toggle, model_pt, src_txt, n_best, beam_sz, gpu_idx],
        outputs=[nmr2s_log, nmr2s_list],
    )

    series_btn.click(
        run_timeseries,
        inputs=[series_species, series_files],
        outputs=[series_log, series_df, series_json],
    )

if __name__ == "__main__":
    demo.launch(share=True)
