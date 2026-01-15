# app/gradio_llm_app.py
"""
LLM-Enhanced Gradio UI for NMR Chemistry Analysis.

Features:
- LLM Agent Analysis: Natural language input for chemistry analysis
- Manual Deconvolution: Step-by-step control over the analysis pipeline
- Time-Series Analysis: Track reactions over time with multiple spectra
"""

import os
import re
import json
import tempfile
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import gradio as gr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Tool imports
from .tools_nmrbank import get_reference_by_smiles
from .tools_reactiont5 import resolve_names_to_smiles, propose_products, get_unique_components
from .tools_deconvolve import deconvolve_spectra

# Optional tools
try:
    from .tools_magnetstein import (
        quantify_single as magnet_quant_single,
        quantify_timeseries as magnet_quant_series,
    )
    _HAS_MAGNETSTEIN = True
except Exception:
    _HAS_MAGNETSTEIN = False

# LLM support
import requests

# Model configurations: (display_name, api_url, model_id)
LLM_MODEL_CONFIG = {
    # Cerebras (fast Llama models)
    "Llama 3.1 8B (Cerebras)": (
        "https://router.huggingface.co/cerebras/v1/chat/completions",
        "llama3.1-8b"
    ),
    "Llama 3.3 70B (Cerebras)": (
        "https://router.huggingface.co/cerebras/v1/chat/completions",
        "llama-3.3-70b"
    ),
    # DeepSeek
    "DeepSeek V3": (
        "https://router.huggingface.co/sambanova/v1/chat/completions",
        "DeepSeek-V3-0324"
    ),
    # Qwen
    "Qwen 2.5 72B": (
        "https://router.huggingface.co/hyperbolic/v1/chat/completions",
        "Qwen/Qwen2.5-72B-Instruct"
    ),
}

LLM_MODELS = list(LLM_MODEL_CONFIG.keys())


# ----------------------------
# LLM Client
# ----------------------------
class ChemistryLLM:
    """LLM client for parsing chemistry requests via HuggingFace Router."""

    def __init__(self, model_name: str = "Llama 3.1 8B (Cerebras)"):
        self.model_name = model_name
        config = LLM_MODEL_CONFIG.get(model_name, LLM_MODEL_CONFIG["Llama 3.1 8B (Cerebras)"])
        self.api_url, self.model_id = config
        self.headers = {
            "Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}",
        }

    def query(self, user_input: str) -> Dict[str, Any]:
        """Parse chemistry request using LLM."""
        system_prompt = """You are a chemistry analysis expert. Parse the user's request and extract:
1. Chemical reactants (names or SMILES)
2. Reagents/conditions
3. Analysis type (reaction, quantification, timeseries)

Respond with JSON only:
{
    "reactants": ["compound1", "compound2"],
    "reagents": "conditions or reagents",
    "analysis_type": "reaction|quantification|timeseries",
    "notes": "any relevant details"
}

Common chemicals: anisole, benzene, toluene, phenol, bromobenzene, p-bromoanisole, o-bromoanisole, bromine (Br2), iron(III) bromide (FeBr3), pinene, benzyl benzoate, acetic acid, ethanol."""

        try:
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                "temperature": 0.1,
                "max_tokens": 500
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"LLM query failed: {e}")

        return None


# ----------------------------
# Utilities
# ----------------------------
def parse_csv(file) -> Dict[str, List[float]]:
    """Parse CSV with two columns: ppm, intensity."""
    if file is None:
        return {"ppm": [], "intensity": []}
    
    filepath = file.name if hasattr(file, "name") else file
    
    # Try reading with header first (most common case)
    df = pd.read_csv(filepath)
    
    # Check if we got valid numeric data
    # If first column values can't be converted to float, try without header
    try:
        first_val = df.iloc[0, 0]
        float(first_val)
    except (ValueError, TypeError):
        # First row looks like data, not a header - re-read without header
        df = pd.read_csv(filepath, header=None)
    
    if len(df.columns) < 2:
        raise ValueError(f"CSV must have at least 2 columns, got {len(df.columns)}")
    
    df = df.rename(columns={df.columns[0]: "ppm", df.columns[1]: "intensity"})
    return {
        "ppm": df["ppm"].astype(float).tolist(),
        "intensity": df["intensity"].astype(float).tolist(),
    }


def parse_timeseries(files: List) -> Tuple[List[float], List[Dict[str, List[float]]]]:
    """Parse multiple CSVs; extract time from filename."""
    items = []
    num_re = re.compile(r"(\d+(?:\.\d+)?)")
    for f in files or []:
        name = getattr(f, "name", "t0.csv")
        m = num_re.search(os.path.basename(name))
        t = float(m.group(1)) if m else 0.0
        items.append((t, parse_csv(f)))
    items.sort(key=lambda x: x[0])
    return [t for t, _ in items], [m for _, m in items]


def _plot_spectrum(ppm: List[float], intensity: List[float], title: str, style="auto") -> str:
    """Plot spectrum and save to temp PNG."""
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=150)

    if style == "sticks" or (style == "auto" and len(ppm) <= 100):
        for x, y in zip(ppm, intensity or [1.0] * len(ppm)):
            ax.vlines(x, 0, y, linewidth=1.2)
    else:
        ax.plot(ppm, intensity, linewidth=0.8)

    if len(ppm) >= 2:
        ax.invert_xaxis()
    ax.set_xlabel("ppm")
    ax.set_ylabel("intensity")
    ax.set_title(title, fontsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    path = os.path.join(tempfile.mkdtemp(), f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', title)[:40]}.png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def load_references(smiles_list: List[str]) -> List[Dict[str, Any]]:
    """Load reference spectra for list of SMILES."""
    refs = []
    for smiles in smiles_list:
        ref = get_reference_by_smiles(smiles)
        if ref:
            refs.append({
                "name": ref.get("name", ""),
                "smiles": smiles,
                "ppm": ref.get("ppm", []),
                "intensity": ref.get("intensity", [])
            })
    return refs


def render_reference_pngs(refs: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    """Render reference spectra as PNG images."""
    paths = []
    for r in (refs or [])[:limit]:
        ppm = r.get("ppm") or []
        inten = r.get("intensity") or [1.0] * len(ppm)
        if ppm:
            title = r.get("name") or r.get("smiles") or "reference"
            paths.append(_plot_spectrum(ppm, inten, title, style="sticks"))
    return paths


# ----------------------------
# LLM Agent Functions
# ----------------------------
def parse_llm_request(user_input: str, model_name: str = None) -> Dict[str, Any]:
    """
    Parse natural language chemistry request into structured data.
    Uses LLM if model_name provided, otherwise falls back to keyword matching.
    """
    result = {
        "reactants": [],
        "reagents": "",
        "analysis_type": "reaction",
        "raw_input": user_input,
        "llm_used": False
    }

    # Try LLM parsing first
    if model_name and os.environ.get("HF_TOKEN"):
        llm = ChemistryLLM(model_name)
        parsed = llm.query(user_input)
        if parsed:
            result["reactants"] = parsed.get("reactants", [])
            result["reagents"] = parsed.get("reagents", "")
            result["analysis_type"] = parsed.get("analysis_type", "reaction")
            result["notes"] = parsed.get("notes", "")
            result["llm_used"] = True
            return result

    # Fallback: keyword-based parsing
    text = user_input.lower()

    chemicals = ["anisole", "benzene", "toluene", "phenol", "bromobenzene",
                 "br2", "bromine", "febr3", "pinene", "benzyl benzoate",
                 "acetic acid", "ethanol", "methanol", "water", "chlorine", "aspirin"]

    found = []
    for chem in chemicals:
        if chem in text:
            found.append(chem)

    reagent_keywords = ["febr3", "catalyst", "iron", "acid catalyst"]
    for f in found:
        is_reagent = any(rk in f for rk in reagent_keywords)
        if is_reagent:
            result["reagents"] = f
        else:
            result["reactants"].append(f)

    if "time" in text or "kinetic" in text or "series" in text:
        result["analysis_type"] = "timeseries"
    elif "quantif" in text or "deconvol" in text or "mixture" in text:
        result["analysis_type"] = "quantification"

    return result


def run_llm_analysis(
    user_input: str,
    mixture_csv,
    model_name: str = "deepseek-ai/DeepSeek-V3:together"
) -> Tuple[str, str, pd.DataFrame, str, List[str], pd.DataFrame]:
    """
    Run LLM-guided analysis pipeline.

    Returns: (status, narrative, results_df, workflow_json, ref_images, quant_df)
    """
    if not user_input.strip():
        return ("Please describe your chemistry analysis.", "", pd.DataFrame(), "{}", [], pd.DataFrame())

    # Parse the request using LLM
    parsed = parse_llm_request(user_input, model_name=model_name)

    log_parts = [f"Parsed request: {json.dumps(parsed, indent=2)}"]
    results_data = []
    quant_df = pd.DataFrame()
    ref_images = []

    # Step 1: Resolve reactants
    reactant_smiles = []
    for r in parsed["reactants"]:
        resolved = resolve_names_to_smiles(r)
        if resolved:
            reactant_smiles.extend(resolved)
            results_data.append({"Type": "Reactant", "Name": r, "SMILES": resolved[0]})
            log_parts.append(f"Resolved {r} → {resolved[0]}")

    if not reactant_smiles:
        return ("No valid reactants found in request.", "\n".join(log_parts),
                pd.DataFrame(), json.dumps(parsed), [], pd.DataFrame())

    # Step 2: Predict products
    reactants_str = ".".join(reactant_smiles)
    predictions = propose_products(reactants_str, reagents=parsed["reagents"], n_best=5)
    product_smiles = [s for s, _ in predictions]
    unique_products = get_unique_components(product_smiles)

    for p in unique_products[:5]:
        results_data.append({"Type": "Predicted Product", "Name": "-", "SMILES": p})
    log_parts.append(f"Predicted {len(unique_products)} unique products")

    # Step 3: Find references
    all_species = reactant_smiles + unique_products
    refs = load_references(all_species)

    for ref in refs:
        results_data.append({"Type": "Reference Found", "Name": ref["name"], "SMILES": ref["smiles"]})

    ref_images = render_reference_pngs(refs, limit=6)
    log_parts.append(f"Found {len(refs)} reference spectra")

    # Step 4: Quantify if mixture provided
    mixture = parse_csv(mixture_csv)
    if mixture["ppm"] and refs:
        try:
            result = deconvolve_spectra(
                mixture_ppm=mixture["ppm"],
                mixture_intensity=mixture["intensity"],
                refs=refs,
                quiet=True
            )
            conc = result.get("concentrations", {})
            if conc:
                quant_df = pd.DataFrame([
                    {"Component": k, "Proportion": f"{v:.4f}"}
                    for k, v in sorted(conc.items(), key=lambda x: -x[1])
                ])
                log_parts.append("Quantification complete")
        except Exception as e:
            log_parts.append(f"Quantification error: {e}")

    # Generate narrative
    parsing_method = "🤖 LLM" if parsed.get("llm_used") else "📝 Keyword matching"
    narrative = f"""
## Analysis Summary

**Request:** {user_input}

**Parsing method:** {parsing_method}
**Reactants identified:** {', '.join(parsed['reactants']) or 'None'}
**Reagents/conditions:** {parsed['reagents'] or 'None specified'}

**Products predicted:** {len(unique_products)} unique structures
**References found:** {len(refs)} compounds in database

{"**Quantification:** Complete - see results table" if not quant_df.empty else "**Quantification:** No mixture data provided"}
"""

    results_df = pd.DataFrame(results_data)
    status = "Analysis complete" if results_data else "No results"

    return (status, narrative, results_df, json.dumps(parsed, indent=2), ref_images, quant_df)


# ----------------------------
# Manual Deconvolution Functions
# ----------------------------
def run_proposal(reactants_text: str, reagents_text: str, topk: int):
    """Step 1: Propose products from reactants."""
    reactant_smiles = []
    for r in [x.strip() for x in reactants_text.split(",") if x.strip()]:
        resolved = resolve_names_to_smiles(r)
        if resolved:
            reactant_smiles.extend(resolved)

    if not reactant_smiles:
        return [], [], "No valid reactants found.", pd.DataFrame(), [], []

    reactants_str = ".".join(reactant_smiles)
    predictions = propose_products(reactants_str, reagents=reagents_text, n_best=topk)
    products = [s for s, _ in predictions]
    unique = get_unique_components(products)

    # Load references
    all_species = reactant_smiles + unique
    refs = load_references(all_species)

    msg = f"Reactants: {reactant_smiles}\nProducts: {unique}\nReferences found: {len(refs)}"
    table = pd.DataFrame({
        "Species": all_species,
        "In Library": [any(r["smiles"] == s for r in refs) for s in all_species]
    })
    ref_pngs = render_reference_pngs(refs, limit=8)

    return reactant_smiles, products, msg, table, ref_pngs, refs


def run_quantify(mixture_csv, reactants_list, products_list, refs_list, backend: str):
    """Step 2: Quantify mixture using selected references."""
    mixture = parse_csv(mixture_csv)
    if not mixture["ppm"]:
        return "No mixture CSV provided.", pd.DataFrame(), "{}"

    refs = refs_list or []
    if not refs:
        return "No reference spectra available.", pd.DataFrame(), "{}"

    try:
        if backend == "Magnetstein" and _HAS_MAGNETSTEIN:
            result = magnet_quant_single(mixture["ppm"], mixture["intensity"], refs)
            conc = result.get("concentrations", {})
        else:
            result = deconvolve_spectra(
                mixture_ppm=mixture["ppm"],
                mixture_intensity=mixture["intensity"],
                refs=refs,
                quiet=True
            )
            conc = result.get("concentrations", {})

        df = pd.DataFrame([
            {"Component": k, "Proportion": f"{v:.6f}"}
            for k, v in sorted(conc.items(), key=lambda x: -x[1])
        ])

        overlay = {
            "ppm": mixture["ppm"][:100],  # Truncate for JSON
            "mixture": mixture["intensity"][:100],
        }

        return f"Quantification complete ({backend})", df, json.dumps(overlay)

    except Exception as e:
        return f"Error: {e}", pd.DataFrame(), "{}"


# ----------------------------
# Time-Series Functions
# ----------------------------
def run_timeseries(species_text: str, files):
    """Run time-series quantification."""
    if not _HAS_MAGNETSTEIN:
        return "Magnetstein not available for time-series analysis.", pd.DataFrame(), "{}"

    # Resolve species
    species_smiles = []
    for s in [x.strip() for x in species_text.split(",") if x.strip()]:
        resolved = resolve_names_to_smiles(s)
        if resolved:
            species_smiles.extend(resolved)

    if not species_smiles:
        return "No valid species provided.", pd.DataFrame(), "{}"

    refs = load_references(species_smiles)
    if not refs:
        return "No references found for species.", pd.DataFrame(), "{}"

    times, mixes = parse_timeseries(files)
    if not times:
        return "No time-series CSVs uploaded.", pd.DataFrame(), "{}"

    try:
        result = magnet_quant_series(times, mixes, refs)
        props = result.get("proportions", {})

        df = pd.DataFrame({"Time": result["times"]})
        for k, v in props.items():
            df[k] = v

        return "Time-series analysis complete", df, df.to_json(orient="records")

    except Exception as e:
        return f"Error: {e}", pd.DataFrame(), "{}"


# ----------------------------
# Gradio Interface
# ----------------------------
with gr.Blocks(title="NMR Chemistry Analysis") as demo:
    gr.Markdown("# NMR Chemistry Analysis\nLLM-enhanced tools for reaction prediction and mixture quantification.")

    with gr.Tabs():
        # ============ TAB 1: LLM Agent Analysis ============
        with gr.Tab("🤖 LLM Agent Analysis"):
            gr.Markdown("""
            ### Natural Language Chemistry Analysis
            Describe your chemistry problem in plain English. The agent will:
            1. **Parse** your request using an LLM to identify reactants and conditions
            2. **Predict** reaction products using ReactionT5
            3. **Find** NMR reference spectra in the database
            4. **Quantify** your mixture (if spectrum provided)

            *Requires `HF_TOKEN` environment variable for LLM parsing.*
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    llm_input = gr.Textbox(
                        label="Describe your chemistry request",
                        placeholder="Example: I'm brominating anisole with Br2 and FeBr3 catalyst. What products should I expect?",
                        lines=3
                    )
                    llm_mixture = gr.File(label="Upload mixture NMR spectrum (CSV: ppm, intensity)", file_types=[".csv"])
                    llm_model = gr.Dropdown(
                        choices=LLM_MODELS,
                        value=LLM_MODELS[0],
                        label="LLM Model"
                    )
                    llm_btn = gr.Button("🚀 Run LLM Analysis", variant="primary")

                with gr.Column(scale=1):
                    llm_status = gr.Textbox(label="Status", interactive=False)
                    hf_token_status = gr.Markdown(
                        f"**HF_TOKEN:** {'✅ Set' if os.environ.get('HF_TOKEN') else '❌ Not set (fallback to keyword parsing)'}"
                    )

            llm_narrative = gr.Markdown(label="Analysis Narrative")

            with gr.Row():
                llm_results = gr.Dataframe(label="Results", interactive=False)
                llm_quant = gr.Dataframe(label="Quantification", interactive=False)

            llm_gallery = gr.Gallery(label="Reference Spectra", columns=3, height=300)
            llm_workflow = gr.Code(label="Parsed Request (JSON)", language="json")

            llm_btn.click(
                run_llm_analysis,
                inputs=[llm_input, llm_mixture, llm_model],
                outputs=[llm_status, llm_narrative, llm_results, llm_workflow, llm_gallery, llm_quant]
            )

        # ============ TAB 2: Manual Deconvolution ============
        with gr.Tab("🧪 Manual Deconvolution"):
            gr.Markdown("""
            ### Step-by-Step Analysis
            Control each step of the analysis pipeline manually.
            """)

            with gr.Row():
                reactants = gr.Textbox(
                    label="Reactants (comma-separated)",
                    placeholder="anisole, Br2"
                )
                reagents = gr.Textbox(
                    label="Reagents/conditions",
                    placeholder="FeBr3"
                )
                topk = gr.Slider(1, 10, value=5, step=1, label="Top-k products")

            propose_btn = gr.Button("Step 1: Propose Products")

            # State variables
            rxs_state = gr.State([])
            prods_state = gr.State([])
            refs_state = gr.State([])

            proposal_log = gr.Textbox(label="Log", interactive=False, lines=3)
            proposal_table = gr.Dataframe(label="Species", interactive=False)
            proposal_gallery = gr.Gallery(label="Reference Spectra", columns=2, height=300)

            gr.Markdown("---")

            with gr.Row():
                mixture_csv = gr.File(label="Mixture CSV (ppm, intensity)", file_types=[".csv"])
                backend = gr.Dropdown(
                    choices=["Masserstein"] + (["Magnetstein"] if _HAS_MAGNETSTEIN else []),
                    value="Masserstein",
                    label="Backend"
                )

            quantify_btn = gr.Button("Step 2: Quantify Mixture")

            quant_status = gr.Textbox(label="Status", interactive=False)
            quant_table = gr.Dataframe(label="Concentrations", interactive=False)

            propose_btn.click(
                run_proposal,
                inputs=[reactants, reagents, topk],
                outputs=[rxs_state, prods_state, proposal_log, proposal_table, proposal_gallery, refs_state]
            )

            quantify_btn.click(
                run_quantify,
                inputs=[mixture_csv, rxs_state, prods_state, refs_state, backend],
                outputs=[quant_status, quant_table, gr.Textbox(visible=False)]
            )

        # ============ TAB 3: Time-Series ============
        with gr.Tab("📈 Time-Series Analysis"):
            gr.Markdown("""
            ### Reaction Kinetics
            Track component concentrations over time using multiple NMR spectra.

            **Instructions:**
            1. Enter the species you expect (reactants + products)
            2. Upload multiple CSV files named with time points (e.g., `t0.csv`, `t5.csv`, `10min.csv`)
            3. The system will extract times from filenames and track concentrations
            """)

            if not _HAS_MAGNETSTEIN:
                gr.Markdown("⚠️ **Note:** Magnetstein not available. Time-series analysis requires the magnetstein module.")

            series_species = gr.Textbox(
                label="Species to track (comma-separated)",
                placeholder="anisole, p-bromoanisole, o-bromoanisole"
            )
            series_files = gr.File(
                file_count="multiple",
                label="Time-series CSV files",
                file_types=[".csv"]
            )
            series_btn = gr.Button("Analyze Time-Series", variant="primary")

            series_status = gr.Textbox(label="Status", interactive=False)
            series_table = gr.Dataframe(label="Proportions vs Time", interactive=False)
            series_json = gr.Code(label="Data (JSON)", language="json")

            series_btn.click(
                run_timeseries,
                inputs=[series_species, series_files],
                outputs=[series_status, series_table, series_json]
            )


if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7667)
