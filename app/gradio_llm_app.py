# app/gradio_llm_app.py
# ---------------------------------------
# LLM-Enhanced Gradio UI for Chemistry Analysis
# Uses natural language input to orchestrate NMR analysis tools
# ---------------------------------------
from __future__ import annotations

import os, re, json, warnings, tempfile
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import gradio as gr

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Suppress HuggingFace tokenizers parallelism warning
warnings.filterwarnings("ignore", message=".*tokenizers.*parallelism.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- our app modules ---
from .agent_pocketflow import build_llm_graph
from .llm_agent import get_agent
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

try:
    from .tools_nmr2structure import predict_structures_from_token_lines
    _HAS_NMR2STRUCT = True
except Exception:
    _HAS_NMR2STRUCT = False


# ----------------------------
# Utilities
# ----------------------------
def parse_csv(file) -> Dict[str, List[float]]:
    """expects CSV with two columns: ppm,intensity"""
    if file is None:
        return {"ppm": [], "intensity": []}
    df = pd.read_csv(file.name if hasattr(file, "name") else file, header=None)
    if len(df.columns) < 2:
        # try with headers present
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


def _plot_sticks_to_png(ppm: List[float], intensity: List[float], title: str) -> str:
    fig, ax = plt.subplots(figsize=(4, 2), dpi=120)
    for x, y in zip(ppm, intensity if intensity else [1.0] * len(ppm)):
        ax.vlines(x, 0, y, linewidth=1.2)
    try:
        ax.invert_xaxis()
    except Exception:
        pass
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("ppm")
    ax.set_ylabel("intensity")
    fig.tight_layout()
    path = os.path.join(tempfile.gettempdir(), f"{re.sub(r'[^A-Za-z0-9_.-]+','_',title)}_sticks.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_curve_to_png(ppm: List[float], intensity: List[float], title: str) -> str:
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=150)
    ax.plot(ppm, intensity, linewidth=1.0)
    try:
        ax.invert_xaxis()
    except Exception:
        pass
    ax.set_xlabel("ppm")
    ax.set_ylabel("intensity")
    ax.set_title(title, fontsize=10)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    outdir = tempfile.mkdtemp(prefix="nmr_refs_")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", title)[:60]
    path = os.path.join(outdir, f"{safe}.png")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_reference_pngs(refs: List[Dict[str, Any]], limit: int = 8) -> List[str]:
    """Render refs as stick spectra PNGs for the gallery."""
    paths = []
    for r in (refs or [])[:limit]:
        ppm = r.get("ppm") or []
        inten = r.get("intensity") or [1.0] * len(ppm)
        if ppm:
            title = r.get("name") or r.get("smiles") or "reference"
            paths.append(_plot_sticks_to_png(ppm, inten, title))
    return paths


def _read_xy_gradio(file) -> Dict[str, List[float]]:
    df = pd.read_csv(file.name if hasattr(file, "name") else file, header=None, names=["ppm", "intensity"])
    return {"ppm": df["ppm"].astype(float).tolist(),
            "intensity": df["intensity"].astype(float).tolist()}


# ----------------------------
# LLM-Enhanced Analysis Pipeline
# ----------------------------
def run_llm_analysis(
    user_input: str,
    mixture_csv,
    model_name: str = "deepseek-ai/DeepSeek-V3:together",
) -> Tuple[str, str, pd.DataFrame, str, str, list, pd.DataFrame, str]:
    """
    Returns:
      status_message, narrative, results_table, workflow_json, refs_status, refs_pngs
    """
    if not user_input.strip():
        return (
            "Please provide a description of your chemistry analysis request.",
            "",
            pd.DataFrame(),
            "{}",
            "No references yet.",
            [],
        )

    # Parse mixture data if provided
    mixture_data = None
    if mixture_csv is not None:
        mixture_data = parse_csv(mixture_csv)
        if not mixture_data["ppm"]:
            return (
                "Invalid mixture CSV file. Please provide two columns: ppm, intensity.",
                "",
                pd.DataFrame(),
                "{}",
                "No references yet.",
                [],
            )

    try:
        # Build and run the LLM graph — force Masserstein/Gurobi
        graph = build_llm_graph(model_name=model_name, backend="deconvolve")

        # Prepare context
        ctx: Dict[str, Any] = {"user_input": user_input, "mixture": mixture_data}

        # Execute the graph; if it errors, we'll fall back
        try:
            graph.run(ctx)
            result_ctx = ctx
        except Exception:
            result_ctx = {}

        # Fallback linear path if graph produced nothing
        if (not result_ctx) or (not result_ctx.get("steps")):
            agent = get_agent()
            task = agent.parse_chemistry_request(user_input)
            res = agent.execute_analysis(task, mixture_data)
            result_ctx = {
                "task": task,
                "steps": res.get("steps", []),
                "narrative": res.get("narrative", ""),
                "reactants": task.reactants,
                "products": [],
                "refs": [],
                "backend_choice": "deconvolve",
            }
            for st in result_ctx["steps"]:
                if st.get("step") == "propose_products":
                    result_ctx["products"] = st.get("output") or []
                if st.get("step") == "load_references":
                    result_ctx["refs"] = st.get("output") or []
            q = res.get("final_results", {}).get("quantification")
            if q is not None:
                result_ctx["quantification"] = q

        # Extract results (graph or fallback)
        status = "Analysis completed successfully!"
        if "error" in result_ctx:
            status = f"Analysis failed: {result_ctx['error']}"
        narrative = result_ctx.get("narrative", "No narrative generated.")
        steps = result_ctx.get("steps", [])
        # --- ensure we actually have the quantification payload ---
        quant = result_ctx.get("quantification") or {}
        if not quant:
            # Look inside the 'quantify' step output if the graph put it there
            for st in steps:
                if st.get("step") == "quantify":
                    qout = st.get("output") or {}
                    if qout:
                        quant = qout
                        break
        # Persist discovered quant back into context for consistency
        if quant and not result_ctx.get("quantification"):
            result_ctx["quantification"] = quant


        # ---------- Results table (quant FIRST) ----------
        results_data: List[Dict[str, Any]] = []
        reactants = result_ctx.get("reactants", [])
        products  = result_ctx.get("products", [])
        refs      = result_ctx.get("refs", [])

        # Quantification (Masserstein/Gurobi shape preferred)
        quant_rows: List[Dict[str, Any]] = []
        conc: Dict[str, float] = {}
        if quant:
            if "components" in quant:  # ASICS shape (unlikely now, but safe)
                conc = {c["name"]: c["fraction"] for c in quant.get("components", [])}
            elif "concentrations" in quant:
                conc = quant.get("concentrations", {})
            for name, val in sorted(conc.items(), key=lambda kv: -kv[1]):
                quant_rows.append({
                    "Component": name,
                    "Type": "Concentration (deconvolve)",
                    "Value": f"{val:.4f}",
                })
            if quant_rows:
                quant_rows.insert(0, {
                    "Component": "Quant backend",
                    "Type": "Info",
                    "Value": result_ctx.get("backend_choice", "deconvolve"),
                })

        # Build a dedicated quantification table for the UI
        if conc:
            quant_df = pd.DataFrame(
                {"component": list(conc.keys()), "proportion": list(conc.values())}
            ).sort_values("proportion", ascending=False)
        else:
            quant_df = pd.DataFrame()

        # Try to surface solver/stdout text if the backend provided it
        solver_text = ""
        if isinstance(quant, dict):
            solver_text = (
                quant.get("stdout", "") or
                (quant.get("raw", {}).get("stdout", "") if isinstance(quant.get("raw"), dict) else "")
            )

        # Parsed request
        task = result_ctx.get("task")
        parsed_rows: List[Dict[str, Any]] = []
        if task:
            parsed_rows.append({
                "Component": "Parsed Request",
                "Type": "Input Analysis",
                "Value": f"Reactants: {task.reactants}, Analysis: {task.analysis_type}",
            })

        # Reactants & products
        detail_rows: List[Dict[str, Any]] = []
        for i, r in enumerate(reactants):
            detail_rows.append({"Component": f"Reactant {i+1}", "Type": "Input Species", "Value": r})
        for i, p in enumerate(products):
            detail_rows.append({"Component": f"Product {i+1}", "Type": "Predicted Species", "Value": p})

        # Reference spectra found
        for ref in refs:
            detail_rows.append({"Component": ref["name"], "Type": "Reference Found", "Value": ref["smiles"]})

        # Compose with quant rows first
        results_df = pd.DataFrame(quant_rows + parsed_rows + detail_rows)

        # Workflow JSON
        workflow_info = {
            "steps_completed": [step["step"] for step in steps if step["status"] == "completed"],
            "backend_used": result_ctx.get("backend_choice", "unknown"),
            "species_analyzed": len(reactants) + len(products),
            "references_found": len(refs),
            "has_quantification": bool(quant),
        }

        # Refs status + PNG gallery
        if refs:
            names = [r.get("name") or r.get("smiles") for r in refs]
            ref_msg = f"References found: {len(refs)} -> {names}. Plotted: "
        else:
            ref_msg = "No reference spectra found."
        pngs = render_reference_pngs(refs, limit=8)
        if refs:
            ref_msg += f"{len(pngs)} PNG(s)."

        return (
            status,
            narrative,
            results_df,
            json.dumps(workflow_info, indent=2),
            ref_msg,
            pngs,
            quant_df,        # NEW
            solver_text,     # NEW
        )

    except Exception as e:
        error_msg = f"LLM Analysis failed: {str(e)}"
        return error_msg, error_msg, pd.DataFrame(), "{}", "No references (error).", [], pd.DataFrame(), ""


# ----------------------------
# Time-series Analysis (Magnetstein)
# ----------------------------
def run_llm_timeseries(
    user_input: str,
    files,
    model_name: str = "deepseek-ai/DeepSeek-V3:together",
) -> Tuple[str, str, pd.DataFrame, str]:
    if not _HAS_MAGNETSTEIN:
        return "Magnetstein not available for time-series analysis.", "", pd.DataFrame(), "{}"

    if not user_input.strip():
        return (
            "Please describe what species you want to analyze over time.",
            "",
            pd.DataFrame(),
            "{}",
        )

    agent = get_agent()
    task = agent.parse_chemistry_request(user_input)
    if not task.reactants:
        return "Could not identify chemical species from your request.", "", pd.DataFrame(), "{}"

    times, mixes = parse_timeseries(files)
    if not times:
        return "No valid time-series CSV files uploaded.", "", pd.DataFrame(), "{}"

    try:
        from .agent import normalize_smiles_list, load_refs_for_species

        species = normalize_smiles_list(", ".join(task.reactants))
        if not species:
            return "Could not convert species names to valid SMILES.", "", pd.DataFrame(), "{}"

        refs = load_refs_for_species(species)
        if not refs:
            return "No reference spectra found for the specified species.", "", pd.DataFrame(), "{}"

        res = magnet_quant_series(times, mixes, refs)

        prop = res["proportions"]  # dict: name -> series
        df = pd.DataFrame({"time": res["times"]})
        for k, v in prop.items():
            df[k] = v

        narrative = f"""Time-series analysis completed for {len(species)} species over {len(times)} time points.
Species analyzed: {', '.join([r['name'] for r in refs])}
Time range: {min(times):.1f} to {max(times):.1f}
Backend: Magnetstein (robust optimal transport)"""

        workflow_info = {
            "analysis_type": "timeseries",
            "species_count": len(species),
            "time_points": len(times),
            "backend": "magnetstein",
        }

        return "Time-series analysis completed!", narrative, df, json.dumps(workflow_info, indent=2)

    except Exception as e:
        return f"Time-series analysis failed: {str(e)}", "", pd.DataFrame(), "{}"


# ----------------------------
# Manual Deconvolution (Masserstein)
# ----------------------------
def run_manual_deconvolution(
    mixture_file,
    component_files,
    names_text,
    protons_text,
    threads,
    time_limit,
):
    if not mixture_file:
        return "Please upload a mixture CSV.", pd.DataFrame(), ""

    mix = _read_xy_gradio(mixture_file)
    if not mix["ppm"]:
        return "Mixture CSV is empty or invalid.", pd.DataFrame(), ""

    files = component_files or []
    if not files:
        return "Upload at least one component CSV.", pd.DataFrame(), ""

    refs = []
    for f in files:
        xy = _read_xy_gradio(f)
        refs.append(
            {
                "name": os.path.basename(getattr(f, "name", "component.csv")),
                "ppm": xy["ppm"],
                "intensity": xy["intensity"],
            }
        )

    names = [s.strip() for s in (names_text or "").split(",") if s.strip()] or None

    protons = None
    if protons_text and protons_text.strip():
        try:
            protons = [int(x.strip()) for x in protons_text.split(",") if x.strip()]
        except Exception:
            return "Invalid protons list (must be comma-separated integers).", pd.DataFrame(), ""

    try:
        res = deconvolve_spectra(
            mixture_ppm=mix["ppm"],
            mixture_intensity=mix["intensity"],
            refs=refs,
            names=names,
            protons=protons,
            threads=int(threads or 8),
            time_limit=int(time_limit or 300),
            quiet=True,
        )
    except Exception as e:
        return f"Deconvolution failed: {e}", pd.DataFrame(), ""

    conc = res.get("concentrations", {})
    df = pd.DataFrame(
        {"component": list(conc.keys()), "proportion": list(conc.values())}
    ).sort_values("proportion", ascending=False)
    stdout = res.get("stdout", "")
    msg = "Done." if conc else "No concentrations parsed (check stdout)."
    return msg, df, stdout


# ----------------------------
# Gradio Interface
# ----------------------------
with gr.Blocks(title="LLM-Enhanced NMR Chemistry Analysis", theme=gr.themes.Soft()) as demo:

    gr.Markdown(
        """
    # 🧪 LLM-Enhanced NMR Chemistry Analysis

    **Describe your chemistry analysis in natural language!** This tool uses an LLM agent to understand your request and orchestrate specialized chemistry tools.
    """
    )

    with gr.Tabs():
        # --- Tab 1: LLM Agent Analysis ---
        with gr.Tab("🤖 LLM Agent Analysis"):
            gr.Markdown("### Natural Language Chemistry Analysis")

            with gr.Row():
                with gr.Column(scale=3):
                    user_input = gr.Textbox(
                        label="Describe your chemistry analysis request",
                        placeholder="Example: Analyze bromination of anisole with Br2/FeBr3; identify products and quantify.",
                        lines=4,
                    )
                    mixture_file = gr.File(
                        label="Upload mixture NMR spectrum (CSV: ppm,intensity)", file_types=[".csv"]
                    )
                    model_choice = gr.Dropdown(
                        choices=[
                            "deepseek-ai/DeepSeek-V3:together",
                            "meta-llama/Llama-3.1-8B-Instruct",
                            "mistralai/Mixtral-8x7B-Instruct-v0.1",
                        ],
                        value="deepseek-ai/DeepSeek-V3:together",
                        label="LLM Model",
                    )
                    analyze_btn = gr.Button("🚀 Run LLM Analysis", variant="primary", size="lg")

                with gr.Column(scale=2):
                    status_box = gr.Textbox(label="Status", interactive=False, lines=2)
                    workflow_info = gr.JSON(label="Workflow Information")

            with gr.Row():
                with gr.Column():
                    narrative_box = gr.Textbox(
                        label="🎯 Analysis Narrative (Generated by LLM)", interactive=False, lines=6
                    )
                with gr.Column():
                    results_table = gr.Dataframe(label="📊 Analysis Results", interactive=False, wrap=True)

            refs_status = gr.Textbox(label="Reference Spectra Status", interactive=False, lines=2)
            refs_gallery = gr.Gallery(
                label="🖼️ Reference Spectra (PNG)", columns=2, height=320, preview=True
            )
            # NEW: quantification artifacts for deconvolution path
            quant_table = gr.Dataframe(label="Estimated proportions (deconvolve)", interactive=False)
            quant_stdout = gr.Textbox(label="Solver stdout (debug)", interactive=False, lines=6, visible=False)

        # --- Tab 2: Manual Deconvolution ---
        with gr.Tab("🧮 Manual Deconvolution (Masserstein)"):
            gr.Markdown(
                "Upload a mixture CSV and one or more component CSVs (each two columns: ppm,intensity)."
            )
            mixture_csv2 = gr.File(label="Mixture CSV (ppm,intensity)")
            component_csvs = gr.File(file_count="multiple", label="Component CSVs (one per reference)")
            names_text = gr.Textbox(
                label="Component names (comma-separated, optional)",
                placeholder="e.g., alpha-pinene, benzyl benzoate",
            )
            protons_text = gr.Textbox(
                label="Protons per component (comma-separated, optional)", placeholder="e.g., 10, 10"
            )
            with gr.Row():
                threads = gr.Number(value=8, precision=0, label="Threads")
                time_limit = gr.Number(value=300, precision=0, label="Time limit (s)")
            run_deconv_btn = gr.Button("Run Masserstein Deconvolution", variant="primary")
            deconv_status = gr.Textbox(label="Status", interactive=False)
            deconv_table = gr.Dataframe(label="Estimated proportions", interactive=False)
            deconv_stdout = gr.Textbox(label="Solver stdout (debug)", interactive=False, lines=8)

        # --- Tab 3: Time-Series ---
        with gr.Tab("⏱️ Time-Series Analysis"):
            gr.Markdown("Upload multiple CSV files named with time points (e.g., `t0.csv`, `t5.csv`, `t10.csv`).")
            timeseries_input = gr.Textbox(
                label="Describe the species to track over time",
                placeholder="Example: Track p-bromoanisole and o-bromoanisole formation from anisole bromination",
                lines=3,
            )
            timeseries_files = gr.File(
                file_count="multiple", label="Upload time-series CSV files (ppm,intensity)", file_types=[".csv"]
            )
            timeseries_model = gr.Dropdown(
                choices=[
                    "deepseek-ai/DeepSeek-V3:together",
                    "meta-llama/Llama-3.1-8B-Instruct",
                ],
                value="deepseek-ai/DeepSeek-V3:together",
                label="LLM Model",
            )
            timeseries_btn = gr.Button("📈 Run Time-Series Analysis", variant="primary")
            with gr.Row():
                timeseries_status = gr.Textbox(label="Status", interactive=False)
                timeseries_workflow = gr.JSON(label="Workflow Info")
            timeseries_narrative = gr.Textbox(label="Analysis Summary", interactive=False, lines=4)
            timeseries_results = gr.Dataframe(label="Time-Series Results", interactive=False)

        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
            ## How It Works

            This application combines the power of Large Language Models with specialized chemistry tools:
            1. **Parse Natural Language** → understand your request
            2. **Predict Products** → ReactionT5
            3. **Find References** → NMRBank
            4. **Quantify Mixture** → ASICS / Magnetstein / Masserstein
            5. **Generate Narrative** → LLM summarization
            """
            )

    # Wire events
    analyze_btn.click(
        run_llm_analysis,
        inputs=[user_input, mixture_file, model_choice],
        outputs=[
            status_box,
            narrative_box,
            results_table,
            workflow_info,
            refs_status,
            refs_gallery,
            quant_table,     # NEW
            quant_stdout,    # NEW
        ],
    )

    run_deconv_btn.click(
        run_manual_deconvolution,
        inputs=[mixture_csv2, component_csvs, names_text, protons_text, threads, time_limit],
        outputs=[deconv_status, deconv_table, deconv_stdout],
    )

    timeseries_btn.click(
        run_llm_timeseries,
        inputs=[timeseries_input, timeseries_files, timeseries_model],
        outputs=[timeseries_status, timeseries_narrative, timeseries_results, timeseries_workflow],
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="localhost", server_port=7667)
