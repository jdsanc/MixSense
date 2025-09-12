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
                # Execute and ignore return value; flow uses actions, not ctx
                self._fn(ctx)
                return None  # default transition
            except Exception as exc:
                # Store error on ctx and continue; downstream node will handle
                ctx["error"] = str(exc)
                return None
    
    def parse_natural_language(ctx):
        """Parse natural language input using LLM"""
        user_input = ctx.get("user_input", "")
        if not user_input:
            ctx["error"] = "No user input provided"
            return ctx
            
        # Use LLM to parse the request
        task = llm_agent.parse_chemistry_request(user_input)
        ctx["task"] = task
        ctx["reactants"] = task.reactants
        ctx["reagents"] = task.reagents
        ctx["analysis_type"] = task.analysis_type
        ctx["backend_choice"] = task.backend_preference if task.backend_preference != "auto" else backend
        ctx["steps"] = [{"step": "parse_nl", "status": "completed", "output": task}]
        return ctx

    def normalize_and_propose(ctx):
        """Normalize SMILES and propose products"""
        if "error" in ctx:
            return ctx
            
        # Normalize reactants
        if ctx["reactants"]:
            normalized_reactants = normalize_smiles_list(", ".join(ctx["reactants"]))
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
            "input": {"raw_reactants": ctx.get("task").reactants, "reagents": ctx.get("reagents", "")},
            "output": {"normalized_reactants": normalized_reactants, "products": products}
        })
        return ctx

    def retrieve_references(ctx):
        """Retrieve reference spectra from NMRBank"""
        if "error" in ctx:
            return ctx
            
        species = ctx.get("reactants", []) + ctx.get("products", [])
        refs = []
        if species:
            refs = load_refs_for_species(species)
        
        ctx["refs"] = refs
        ctx["steps"].append({
            "step": "retrieve_refs",
            "status": "completed", 
            "input": species,
            "output": [{"name": r["name"], "smiles": r["smiles"]} for r in refs]
        })
        return ctx

    def quantify_mixture(ctx):
        """Quantify mixture using selected backend"""
        if "error" in ctx or not ctx.get("mixture"):
            if not ctx.get("mixture"):
                ctx["steps"].append({
                    "step": "quantify",
                    "status": "skipped",
                    "reason": "No mixture data provided"
                })
            return ctx
            
        mixture = ctx["mixture"]  # {"ppm": [...], "intensity": [...]}
        refs = ctx.get("refs", [])
        backend_choice = ctx.get("backend_choice", "asics")
        
        if not refs:
            ctx["error"] = "No reference spectra available for quantification"
            return ctx
        
        try:
            if backend_choice == "magnetstein":
                result = magnet_quant(mixture["ppm"], mixture["intensity"], refs)
            else:
                result = asics_quantify(
                    crude_ppm=mixture["ppm"],
                    crude_intensity=mixture["intensity"],
                    refs=[{"name": r["name"], "ppm": r["ppm"], "intensity": r["intensity"]} for r in refs],
                    max_shift=0.02,
                    quant_method="FWER"
                )
            
            ctx["quantification"] = result
            ctx["steps"].append({
                "step": "quantify",
                "status": "completed",
                "input": {"backend": backend_choice, "num_refs": len(refs)},
                "output": result
            })
        except Exception as e:
            ctx["error"] = f"Quantification failed: {str(e)}"
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
            return ctx
            
        # Prepare summary for LLM
        task = ctx.get("task")
        steps = ctx.get("steps", [])
        
        summary_parts = []
        for step in steps:
            if step["status"] == "completed":
                if step["step"] == "normalize_propose":
                    reactants = step["output"]["normalized_reactants"]
                    products = step["output"]["products"]
                    if reactants:
                        summary_parts.append(f"Identified reactants: {reactants}")
                    if products:
                        summary_parts.append(f"Proposed products: {products}")
                elif step["step"] == "retrieve_refs":
                    refs_found = [r["name"] for r in step["output"]]
                    if refs_found:
                        summary_parts.append(f"Found reference spectra for: {refs_found}")
                elif step["step"] == "quantify":
                    backend = step["input"]["backend"]
                    summary_parts.append(f"Performed quantification using {backend}")
        
        system_prompt = """You are a chemistry expert. Generate a clear, professional narrative summary of the NMR analysis results. 
Be specific about what was found and what methods were used. Keep it concise but informative."""
        
        user_prompt = f"""
Original request: {task.additional_context if task else 'Chemistry analysis request'}
Analysis steps completed: {'; '.join(summary_parts) if summary_parts else 'Basic parsing completed'}

Please provide a narrative summary of this NMR chemistry analysis.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            narrative = llm_agent._query_llm(messages)
            ctx["narrative"] = narrative if not narrative.startswith("LLM Error") else "Analysis completed successfully."
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
    
    # Chain the nodes using >> operator
    flow = parse_node >> normalize_node >> refs_node >> quantify_node >> narrate_node
    return flow
