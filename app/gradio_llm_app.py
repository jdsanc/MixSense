# app/gradio_llm_app.py
"""
LLM-Enhanced Gradio UI for NMR Chemistry Analysis.

Features:
- LLM Agent Chat: Natural language input with autonomous tool calling
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
from .utils import plot_spectrum

# Agent import
from .chemistry_agent import ChemistryAgent, LLM_MODELS, create_agent

# Optional tools
try:
    from .tools_magnetstein import (
        quantify_single as magnet_quant_single,
        quantify_timeseries as magnet_quant_series,
    )
    _HAS_MAGNETSTEIN = True
except Exception:
    _HAS_MAGNETSTEIN = False


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


# Use shared plot_spectrum from utils (aliased for backward compatibility)
_plot_spectrum = plot_spectrum


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
# Agent Chat Functions
# ----------------------------
# Global agent instance (will be recreated per model change)
_agent: Optional[ChemistryAgent] = None


def get_agent(model_name: str) -> ChemistryAgent:
    """Get or create the agent."""
    global _agent
    if _agent is None or _agent.model_name != model_name:
        _agent = create_agent(model_name)
    return _agent


def format_tool_calls(tool_calls: List[Dict]) -> str:
    """Format tool calls for display."""
    if not tool_calls:
        return ""

    lines = ["**Tool Calls:**"]
    for tc in tool_calls:
        tool = tc.get("tool", "unknown")
        args = tc.get("arguments", {})
        result = tc.get("result", {})

        # Format arguments
        args_str = ", ".join(f"{k}={repr(v)[:50]}" for k, v in args.items())

        # Format result summary
        if "error" in result:
            result_str = f"Error: {result['error']}"
        elif "smiles" in result and "all_matches" not in result:
            result_str = f"SMILES: {result['smiles']}"
        elif "all_matches" in result:
            result_str = f"SMILES: {result['smiles']}"
        elif "predictions" in result:
            unique = result.get('unique_products', [])
            result_str = f"{len(unique)} products: {', '.join(unique[:3])}{'...' if len(unique) > 3 else ''}"
        elif "found" in result:
            result_str = f"Found {result['found']} refs, missing {len(result.get('missing', []))}"
        elif "concentrations" in result:
            conc = result["concentrations"]
            if conc:
                top = sorted(conc.items(), key=lambda x: -x[1])[:3]
                result_str = ", ".join(f"{k}: {v:.2%}" for k, v in top)
            else:
                result_str = "No concentrations found"
        else:
            result_str = str(result)[:100]

        lines.append(f"- `{tool}({args_str})` → {result_str}")

    return "\n".join(lines)


def chat_respond(
    message: str,
    history: List[Dict[str, str]],
    model_name: str,
    mixture_file
) -> Tuple[List[Dict[str, str]], str, str]:
    """
    Handle a chat message with the agent.

    Returns: (updated_history, tool_log, status)
    """
    if history is None:
        history = []

    if not message.strip():
        return history, "", "Please enter a message."

    if not os.environ.get("HF_TOKEN"):
        return history, "", "HF_TOKEN not set. Please set it in your environment."

    # Get agent and configure
    agent = get_agent(model_name)

    # Set mixture data if provided
    if mixture_file is not None:
        try:
            mixture = parse_csv(mixture_file)
            if mixture["ppm"]:
                agent.set_mixture_data(mixture["ppm"], mixture["intensity"])
        except Exception as e:
            pass  # Silently ignore CSV parse errors

    # Run the agent
    all_tool_calls = []
    final_response = ""

    try:
        for response, tool_calls in agent.run(message):
            final_response = response
            all_tool_calls.extend(tool_calls)
    except Exception as e:
        final_response = f"Error: {str(e)}"

    # Collect spectrum images from tool calls
    spectrum_images = []
    for tc in all_tool_calls:
        result = tc.get("result", {})
        # Single image from lookup_nmr_reference
        if "spectrum_image" in result:
            spectrum_images.append(result["spectrum_image"])
        # Multiple images from load_references_for_smiles
        if "spectrum_images" in result:
            spectrum_images.extend(result["spectrum_images"])

    # Update history using messages format (Gradio 5.x)
    # Add user message and text response first
    history = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": final_response}
    ]
    
    # Add spectrum images as separate assistant messages
    # Gradio 5.x format for files: {"path": "...", "alt_text": "..."}
    for img_path in spectrum_images:
        history.append({
            "role": "assistant", 
            "content": {"path": img_path, "alt_text": "NMR Spectrum"}
        })

    # Format tool log
    tool_log = format_tool_calls(all_tool_calls)

    status = f"Completed with {len(all_tool_calls)} tool calls" if all_tool_calls else "Completed"

    return history, tool_log, status


def clear_chat(model_name: str) -> Tuple[List[Dict[str, str]], str, str]:
    """Clear the chat and agent state."""
    agent = get_agent(model_name)
    agent.clear_state()
    return [], "", "Chat cleared"


def save_session(model_name: str) -> str:
    """Export the current session as JSON and return the file path."""
    agent = get_agent(model_name)
    session_data = agent.export_session()
    
    # Save to temp file
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"session_{timestamp}.json"
    filepath = os.path.join(tempfile.mkdtemp(), filename)
    
    with open(filepath, "w") as f:
        json.dump(session_data, f, indent=2, default=str)
    
    return filepath


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
        # ============ TAB 1: LLM Agent Chat ============
        with gr.Tab("Agent Chat"):
            gr.Markdown("""
            ### Autonomous Chemistry Agent
            Chat with an LLM agent that can autonomously:
            - Resolve chemical names to SMILES
            - Predict reaction products
            - Look up NMR reference spectra
            - Quantify mixture compositions

            The agent decides which tools to use based on your request.

            *Requires `HF_TOKEN` environment variable.*
            """)

            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Chat",
                        height=400,
                        type="messages",
                    )

                    with gr.Row():
                        chat_input = gr.Textbox(
                            label="Your message",
                            placeholder="Example: What products do I get from brominating anisole with Br2 and FeBr3?",
                            lines=2,
                            scale=4,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                    with gr.Row():
                        mixture_upload = gr.File(
                            label="Upload mixture spectrum (CSV: ppm, intensity)",
                            file_types=[".csv"],
                        )
                        model_dropdown = gr.Dropdown(
                            choices=LLM_MODELS,
                            value=LLM_MODELS[2],  # DeepSeek V3 as default
                            label="LLM Model",
                        )
                    
                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", variant="secondary")

                with gr.Column(scale=1):
                    chat_status = gr.Textbox(label="Status", interactive=False)
                    hf_token_status = gr.Markdown(
                        f"**HF_TOKEN:** {'Set' if os.environ.get('HF_TOKEN') else 'Not set'}"
                    )
                    tool_log = gr.Markdown(label="Tool Calls")
                    save_btn = gr.DownloadButton("💾 Save Session", variant="secondary")

            # Event handlers
            send_btn.click(
                chat_respond,
                inputs=[chat_input, chatbot, model_dropdown, mixture_upload],
                outputs=[chatbot, tool_log, chat_status],
            ).then(
                lambda: "",
                outputs=[chat_input],
            )

            chat_input.submit(
                chat_respond,
                inputs=[chat_input, chatbot, model_dropdown, mixture_upload],
                outputs=[chatbot, tool_log, chat_status],
            ).then(
                lambda: "",
                outputs=[chat_input],
            )

            clear_btn.click(
                clear_chat,
                inputs=[model_dropdown],
                outputs=[chatbot, tool_log, chat_status],
            )
            
            save_btn.click(
                save_session,
                inputs=[model_dropdown],
                outputs=[save_btn],
            )

        # ============ TAB 2: Manual Deconvolution ============
        with gr.Tab("Manual Deconvolution"):
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
        with gr.Tab("Time-Series Analysis"):
            gr.Markdown("""
            ### Reaction Kinetics
            Track component concentrations over time using multiple NMR spectra.

            **Instructions:**
            1. Enter the species you expect (reactants + products)
            2. Upload multiple CSV files named with time points (e.g., `t0.csv`, `t5.csv`, `10min.csv`)
            3. The system will extract times from filenames and track concentrations
            """)

            if not _HAS_MAGNETSTEIN:
                gr.Markdown("**Note:** Magnetstein not available. Time-series analysis requires the magnetstein module.")

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
