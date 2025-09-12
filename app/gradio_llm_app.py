# app/gradio_llm_app.py
# ---------------------------------------
# LLM-Enhanced Gradio UI for Chemistry Analysis
# Uses natural language input to orchestrate NMR analysis tools
# ---------------------------------------
import os, json, re
import numpy as np
import pandas as pd
import gradio as gr
import warnings

# Suppress HuggingFace tokenizers parallelism warning
warnings.filterwarnings("ignore", message=".*tokenizers.*parallelism.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import List, Dict, Any, Tuple, Optional

# --- our app modules ---
from .agent_pocketflow import build_llm_graph
from .llm_agent import get_agent, process_natural_language_request

# Check tool availability
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
# LLM-Enhanced Analysis Pipeline
# ----------------------------
def run_llm_analysis(
    user_input: str, mixture_csv, model_name: str = "deepseek-ai/DeepSeek-V3:together"
) -> Tuple[str, str, pd.DataFrame, str]:
    """
    Main LLM analysis function that processes natural language input
    Returns: (status_message, narrative, results_table, workflow_json)
    """
    if not user_input.strip():
        return (
            "Please provide a description of your chemistry analysis request.",
            "",
            pd.DataFrame(),
            "{}",
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
            )

    try:
        # Build and run the LLM graph
        graph = build_llm_graph(model_name=model_name, backend="auto")

        # Prepare context
        ctx = {"user_input": user_input, "mixture": mixture_data}

        # Execute the graph
        graph.run(ctx)
        result_ctx = ctx

        # Extract results
        status = "Analysis completed successfully!"
        if "error" in result_ctx:
            status = f"Analysis failed: {result_ctx['error']}"

        narrative = result_ctx.get("narrative", "No narrative generated.")
        steps = result_ctx.get("steps", [])

        # Build results table
        results_data = []

        # Add task information
        task = result_ctx.get("task")
        if task:
            results_data.append(
                {
                    "Component": "Parsed Request",
                    "Type": "Input Analysis",
                    "Value": f"Reactants: {task.reactants}, Analysis: {task.analysis_type}",
                }
            )

        # Add reactants and products
        reactants = result_ctx.get("reactants", [])
        products = result_ctx.get("products", [])

        for i, reactant in enumerate(reactants):
            results_data.append(
                {
                    "Component": f"Reactant {i + 1}",
                    "Type": "Input Species",
                    "Value": reactant,
                }
            )

        for i, product in enumerate(products):
            results_data.append(
                {
                    "Component": f"Product {i + 1}",
                    "Type": "Predicted Species",
                    "Value": product,
                }
            )

        # Add reference spectra info
        refs = result_ctx.get("refs", [])
        for ref in refs:
            results_data.append(
                {
                    "Component": ref["name"],
                    "Type": "Reference Found",
                    "Value": ref["smiles"],
                }
            )

        # Add quantification results
        quant = result_ctx.get("quantification", {})
        if quant:
            if "components" in quant:  # ASICS format
                for comp in quant["components"]:
                    results_data.append(
                        {
                            "Component": comp["name"],
                            "Type": "Concentration",
                            "Value": f"{comp['fraction']:.4f}",
                        }
                    )
            elif "concentrations" in quant:  # Magnetstein format
                for name, conc in quant["concentrations"].items():
                    results_data.append(
                        {
                            "Component": name,
                            "Type": "Concentration",
                            "Value": f"{conc:.4f}",
                        }
                    )

        results_df = pd.DataFrame(results_data)

        # Create workflow JSON
        workflow_info = {
            "steps_completed": [
                step["step"] for step in steps if step["status"] == "completed"
            ],
            "backend_used": result_ctx.get("backend_choice", "unknown"),
            "species_analyzed": len(reactants) + len(products),
            "references_found": len(refs),
            "has_quantification": bool(quant),
        }

        return status, narrative, results_df, json.dumps(workflow_info, indent=2)

    except Exception as e:
        error_msg = f"LLM Analysis failed: {str(e)}"
        return error_msg, error_msg, pd.DataFrame(), "{}"


# ----------------------------
# Time-series Analysis (Enhanced)
# ----------------------------
def run_llm_timeseries(
    user_input: str, files, model_name: str = "deepseek-ai/DeepSeek-V3:together"
) -> Tuple[str, str, pd.DataFrame, str]:
    """LLM-enhanced time-series analysis"""
    if not _HAS_MAGNETSTEIN:
        return (
            "Magnetstein not available for time-series analysis.",
            "",
            pd.DataFrame(),
            "{}",
        )

    if not user_input.strip():
        return (
            "Please describe what species you want to analyze over time.",
            "",
            pd.DataFrame(),
            "{}",
        )

    # Parse the request using LLM
    agent = get_agent()
    task = agent.parse_chemistry_request(user_input)

    if not task.reactants:
        return (
            "Could not identify chemical species from your request.",
            "",
            pd.DataFrame(),
            "{}",
        )

    # Parse time series data
    times, mixes = parse_timeseries(files)
    if not times:
        return "No valid time-series CSV files uploaded.", "", pd.DataFrame(), "{}"

    try:
        # Import here to avoid circular imports
        from .agent import normalize_smiles_list, load_refs_for_species

        # Normalize species names
        species = normalize_smiles_list(", ".join(task.reactants))
        if not species:
            return (
                "Could not convert species names to valid SMILES.",
                "",
                pd.DataFrame(),
                "{}",
            )

        # Get references
        refs = load_refs_for_species(species)
        if not refs:
            return (
                "No reference spectra found for the specified species.",
                "",
                pd.DataFrame(),
                "{}",
            )

        # Run time-series quantification
        from .tools_magnetstein import quantify_timeseries

        res = quantify_timeseries(times, mixes, refs)

        # Build results DataFrame
        prop = res["proportions"]  # dict: name -> series
        df = pd.DataFrame({"time": res["times"]})
        for k, v in prop.items():
            df[k] = v

        # Generate narrative
        narrative = f"""
Time-series analysis completed for {len(species)} species over {len(times)} time points.
Species analyzed: {", ".join([r["name"] for r in refs])}
Time range: {min(times):.1f} to {max(times):.1f}
Backend: Magnetstein (robust optimal transport)
        """.strip()

        workflow_info = {
            "analysis_type": "timeseries",
            "species_count": len(species),
            "time_points": len(times),
            "backend": "magnetstein",
        }

        return (
            "Time-series analysis completed!",
            narrative,
            df,
            json.dumps(workflow_info, indent=2),
        )

    except Exception as e:
        return f"Time-series analysis failed: {str(e)}", "", pd.DataFrame(), "{}"


# ----------------------------
# Gradio Interface
# ----------------------------
with gr.Blocks(
    title="LLM-Enhanced NMR Chemistry Analysis", theme=gr.themes.soft()
) as demo:
    gr.Markdown("""
    # 🧪 LLM-Enhanced NMR Chemistry Analysis
    
    **Describe your chemistry analysis in natural language!** This tool uses an LLM agent to understand your request and orchestrate specialized chemistry tools.
    
    **Examples of what you can ask:**
    - "I want to analyze a bromination reaction of anisole with Br2 and FeBr3"
    - "Help me quantify the products from reacting benzene with bromine"
    - "Analyze the time evolution of p-bromoanisole formation from anisole"
    """)

    with gr.Tabs():
        with gr.Tab("🤖 LLM Agent Analysis"):
            gr.Markdown("### Natural Language Chemistry Analysis")

            with gr.Row():
                with gr.Column(scale=3):
                    user_input = gr.Textbox(
                        label="Describe your chemistry analysis request",
                        placeholder="Example: I want to analyze the bromination of anisole with Br2 using FeBr3 catalyst. Please identify the products and quantify the mixture.",
                        lines=4,
                    )

                    mixture_file = gr.File(
                        label="Upload mixture NMR spectrum (CSV: ppm, intensity)",
                        file_types=[".csv"],
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

                    analyze_btn = gr.Button(
                        "🚀 Run LLM Analysis", variant="primary", size="lg"
                    )

                with gr.Column(scale=2):
                    status_box = gr.Textbox(label="Status", interactive=False, lines=2)
                    workflow_info = gr.JSON(label="Workflow Information")

            with gr.Row():
                with gr.Column():
                    narrative_box = gr.Textbox(
                        label="🎯 Analysis Narrative (Generated by LLM)",
                        interactive=False,
                        lines=6,
                    )

                with gr.Column():
                    results_table = gr.Dataframe(
                        label="📊 Analysis Results", interactive=False, wrap=True
                    )

        with gr.Tab("⏱️ Time-Series Analysis"):
            gr.Markdown("### LLM-Enhanced Time-Series Analysis")
            gr.Markdown(
                "Upload multiple CSV files named with time points (e.g., `t0.csv`, `t5.csv`, `t10.csv`)"
            )

            timeseries_input = gr.Textbox(
                label="Describe the species to track over time",
                placeholder="Example: Track the formation of p-bromoanisole and o-bromoanisole from anisole bromination",
                lines=3,
            )

            timeseries_files = gr.File(
                file_count="multiple",
                label="Upload time-series CSV files (ppm, intensity)",
                file_types=[".csv"],
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

            timeseries_narrative = gr.Textbox(
                label="Analysis Summary", interactive=False, lines=4
            )

            timeseries_results = gr.Dataframe(
                label="Time-Series Results", interactive=False
            )

        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## How It Works
            
            This application combines the power of Large Language Models with specialized chemistry tools:
            
            ### 🧠 LLM Agent Workflow:
            1. **Parse Natural Language**: LLM understands your chemistry request
            2. **Identify Species**: Converts chemical names to SMILES notation  
            3. **Predict Products**: Uses ReactionT5 for product prediction
            4. **Find References**: Searches NMRBank for reference spectra
            5. **Quantify Mixture**: Uses ASICS or Magnetstein for quantification
            6. **Generate Narrative**: LLM creates a summary of results
            
            ### 🔧 Tools Used:
            - **ReactionT5**: Product prediction from reactants
            - **NMRBank**: Reference NMR spectra database
            - **ASICS**: Standard NMR quantification (R-based)
            - **Magnetstein**: Robust quantification using optimal transport
            - **PocketFlow**: Workflow orchestration framework
            
            ### 🎯 Perfect for:
            - Reaction analysis and product identification
            - NMR mixture quantification
            - Time-series kinetic studies
            - Educational chemistry demonstrations
            
            **Note**: Requires `HF_TOKEN` environment variable for LLM access.
            """)

    # Wire up the events
    analyze_btn.click(
        run_llm_analysis,
        inputs=[user_input, mixture_file, model_choice],
        outputs=[status_box, narrative_box, results_table, workflow_info],
    )

    timeseries_btn.click(
        run_llm_timeseries,
        inputs=[timeseries_input, timeseries_files, timeseries_model],
        outputs=[
            timeseries_status,
            timeseries_narrative,
            timeseries_results,
            timeseries_workflow,
        ],
    )

if __name__ == "__main__":
    demo.launch(share=True, server_name="localhost", server_port=7860)
