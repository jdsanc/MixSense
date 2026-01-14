"""
LLM-Enhanced PocketFlow graph for chemistry analysis:
  ParseNL (LLM) -> ProposeProducts (ReactionT5) -> RetrieveRefs (NMRBank) ->
  Quantify (ASICS or Magnetstein) -> (Optional) NMR->Structure -> Narrate (LLM)
"""
from pocketflow import Node, Flow  # pip install pocketflow
from .llm_agent import ChemistryLLMAgent, AnalysisTask
from .tools_reactiont5 import propose_products
from .tools_nmrbank import get_reference_by_smiles
from .tools_asics import asics_quantify
from .tools_magnetstein import quantify_single as magnet_quant
from .agent import normalize_smiles_list, step_propose, load_refs_for_species
import os

from .tools_deconvolve import deconvolve_spectra

def build_llm_graph(model_name="deepseek-ai/DeepSeek-V3:together", backend="auto"):
    """Build an LLM-enhanced PocketFlow graph for chemistry analysis"""

    llm_agent = ChemistryLLMAgent(model_name)

    class FnNode(Node):
        """PocketFlow Node that executes a provided function on the shared ctx.
        The function should mutate the ctx in-place. The node returns the default action.
        """
        def __init__(self, fn, max_retries: int = 1, wait: float = 0):
            super().__init__(max_retries=max_retries, wait=wait)
            self._fn = fn

        def prep(self, shared):
            return shared

        def exec(self, ctx):
            try:
                # Execute function; mutate ctx in-place
                self._fn(ctx)
            except Exception as exc:
                # Store error on ctx and continue; downstream node will handle
                if isinstance(ctx, dict):
                    ctx["error"] = str(exc)
            # Return None to follow default transition semantics
            return None

    # ----------------------------
    # Helpers
    # ----------------------------
    def _ensure_steps(ctx):
        if "steps" not in ctx or not isinstance(ctx["steps"], list):
            ctx["steps"] = []

    # ----------------------------
    # Nodes
    # ----------------------------
    def parse_natural_language(ctx):
        """Parse natural language input using LLM"""
        user_input = ctx.get("user_input", "")
        if not user_input:
            ctx["error"] = "No user input provided"
            return ctx

        # Use LLM to parse the request
        task = llm_agent.parse_chemistry_request(user_input)
        ctx["task"] = task
        ctx["reactants"] = getattr(task, "reactants", []) or []
        ctx["reagents"] = getattr(task, "reagents", "") or ""
        ctx["analysis_type"] = getattr(task, "analysis_type", "") or ""
        backend_pref = getattr(task, "backend_preference", "auto") or "auto"
        ctx["backend_choice"] = backend_pref if backend_pref != "auto" else backend

        # Initialize steps log
        ctx["steps"] = [{"step": "parse_nl", "status": "completed", "output": task}]
        return ctx
        
    
    def normalize_and_propose(ctx):
        """Normalize SMILES and propose products"""
        
        if "error" in ctx:
            return ctx
        _ensure_steps(ctx)



        # Normalize reactants
        raw_reactants = ctx.get("reactants", []) or []
        if raw_reactants:
            normalized_reactants = normalize_smiles_list(", ".join(raw_reactants))
            ctx["reactants"] = normalized_reactants
        else:
            normalized_reactants = []

        # Propose products if we have reactants
        products = []
        if normalized_reactants:
            products = step_propose(normalized_reactants, ctx.get("reagents", ""), topk=5)

        ctx["products"] = products
        ctx["steps"].append({
            "step": "normalize_propose",
            "status": "completed",
            "input": {
                "raw_reactants": getattr(ctx.get("task", None), "reactants", []),
                "reagents": ctx.get("reagents", "")
            },
            "output": {"normalized_reactants": normalized_reactants, "products": products}
        })
        return ctx

    def retrieve_references(ctx):
        """Retrieve reference spectra from NMRBank"""
        if "error" in ctx:
            return ctx
        _ensure_steps(ctx)

        species = (ctx.get("reactants", []) or []) + (ctx.get("products", []) or [])
        refs = load_refs_for_species(species) if species else []

        ctx["refs"] = refs
        ctx["steps"].append({
            "step": "retrieve_refs",
            "status": "completed",
            "input": species,
            "output": [{"name": r["name"], "smiles": r["smiles"]} for r in refs]
        })
        return ctx
    # app/agent_pocketflow.py

    from .tools_deconvolve import deconvolve_spectra  # add at top

    def quantify_mixture(ctx):
        """Quantify mixture using Masserstein (Gurobi) deconvolution only."""
        # Skip if earlier error or no mixture
        if "error" in ctx or not ctx.get("mixture"):
            _ensure_steps(ctx)
            if not ctx.get("mixture"):
                ctx["steps"].append({
                    "step": "quantify",
                    "status": "skipped",
                    "reason": "No mixture data provided"
                })
            return ctx

        mixture = ctx["mixture"]  # {"ppm": [...], "intensity": [...]}
        refs = ctx.get("refs", []) or []
        if not refs:
            ctx["error"] = "No reference spectra available for quantification"
            return ctx

        try:
            # Always use Masserstein deconvolution
            names = [r.get("name", f"comp{i}") for i, r in enumerate(refs)]
            protons = [int(r.get("protons", 1)) for r in refs]

            result = deconvolve_spectra(
                mixture_ppm=mixture["ppm"],
                mixture_intensity=mixture["intensity"],
                refs=[{"name": r["name"], "ppm": r["ppm"], "intensity": r["intensity"]} for r in refs],
                names=names,
                protons=protons,
                threads=int(os.environ.get("DECONV_THREADS", "8")),
                time_limit=int(os.environ.get("DECONV_TIME_LIMIT", "300")),
                quiet=True,
            )

            ctx["backend_choice"] = "deconvolve"
            ctx["quantification"] = result

            _ensure_steps(ctx)
            ctx["steps"].append({
                "step": "quantify",
                "status": "completed",
                "input": {"backend": "deconvolve", "num_refs": len(refs)},
                "output": result
            })
        except Exception as e:
            ctx["error"] = f"Quantification failed (deconvolve): {e}"
            _ensure_steps(ctx)
            ctx["steps"].append({
                "step": "quantify",
                "status": "failed",
                "error": str(e)
            })

        return ctx


    def generate_narrative(ctx):
        """Generate natural language narrative of results using LLM"""
        if "error" in ctx:
            ctx["narrative"] = f"Analysis failed: {ctx['error']}"
            _ensure_steps(ctx)
            ctx["steps"].append({
                "step": "generate_narrative",
                "status": "completed",
                "output": ctx["narrative"]
            })
            return ctx

        _ensure_steps(ctx)

        # Prepare summary for LLM
        task = ctx.get("task")
        steps = ctx.get("steps", [])

        summary_parts = []
        for step in steps:
            if step.get("status") == "completed":
                if step.get("step") == "normalize_propose":
                    reactants = step["output"].get("normalized_reactants", [])
                    products = step["output"].get("products", [])
                    if reactants:
                        summary_parts.append(f"Identified reactants: {reactants}")
                    if products:
                        summary_parts.append(f"Proposed products: {products}")
                elif step.get("step") == "retrieve_refs":
                    refs_found = [r.get("name") for r in step.get("output", [])]
                    if refs_found:
                        summary_parts.append(f"Found reference spectra for: {refs_found}")
                elif step.get("step") == "quantify":
                    backend = step.get("input", {}).get("backend", "unknown")
                    summary_parts.append(f"Performed quantification using {backend}")

        system_prompt = """You are a chemistry expert. Generate a clear, professional narrative summary of the NMR analysis results. 
Be specific about what was found and what methods were used. Keep it concise but informative."""

        user_prompt = f"""
Original request: {getattr(task, 'additional_context', 'Chemistry analysis request') if task else 'Chemistry analysis request'}
Analysis steps completed: {'; '.join(summary_parts) if summary_parts else 'Basic parsing completed'}

Please provide a narrative summary of this NMR chemistry analysis.
""".strip()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            narrative = llm_agent._query_llm(messages)
            ctx["narrative"] = narrative if not str(narrative).startswith("LLM Error") else "Analysis completed successfully."
        except Exception as e:
            ctx["narrative"] = f"Analysis completed. Narrative generation failed: {str(e)}"

        ctx["steps"].append({
            "step": "generate_narrative",
            "status": "completed",
            "output": ctx["narrative"]
        })

        return ctx

    # Build the flow by chaining nodes
    parse_node = FnNode(parse_natural_language)
    normalize_node = FnNode(normalize_and_propose)
    refs_node = FnNode(retrieve_references)
    quantify_node = FnNode(quantify_mixture)
    narrate_node = FnNode(generate_narrative)

    # Chain the nodes using >> operator and explicitly return a Flow starting at the head node
    parse_node >> normalize_node >> refs_node >> quantify_node >> narrate_node
    return Flow(parse_node)
