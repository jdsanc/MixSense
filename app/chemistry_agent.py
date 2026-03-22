"""
Chemistry Agent - True LLM Agent with Tool Calling.

This agent autonomously decides which tools to call, interprets results,
and orchestrates the entire NMR chemistry analysis workflow.
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Generator

from openai import OpenAI

from .tool_definitions import CHEMISTRY_TOOLS
from .tools_nmrbank import get_reference_by_smiles
from .tools_reactiont5 import resolve_names_to_smiles, propose_products, get_unique_components
from .utils import plot_spectrum

# Try to import magnetstein, fall back to masserstein
try:
    from .tools_magnetstein import quantify_single as magnetstein_quantify
    _USE_MAGNETSTEIN = True
except ImportError:
    from .tools_deconvolve import deconvolve_spectra
    _USE_MAGNETSTEIN = False


# System prompt that guides the agent's behavior
SYSTEM_PROMPT = """You are an expert NMR chemistry analysis assistant with access to specialized chemistry tools.

## Rules

### Name → SMILES
You are an expert chemist. Convert chemical names to canonical SMILES directly using your knowledge.
Only call resolve_chemical_name for names you are genuinely unsure about.
Pass the SMILES you determined straight to predict_products.

### Reaction Prediction and NMR Spectra
**NEVER invent reaction products or NMR spectra.** These MUST come from the tools.
If predict_products or lookup_nmr_reference fails, report the failure — do not guess.

## Available Tools

### resolve_chemical_name
Optional: look up a SMILES you are uncertain about.

### predict_products
Predict reaction products using the ReactionT5 ML model.
- Input: reactants (SMILES joined by " . "), reagents (optional)
- Output: `unique_products` list contains the predicted SMILES
- **USE THE RETURNED `unique_products` OR `all_species_for_nmr_lookup` EXACTLY**

### load_references_for_smiles  
Look up NMR spectra for multiple compounds. Pass the `all_species_for_nmr_lookup` from predict_products.

### lookup_nmr_reference
Look up NMR spectrum for a single compound.

### deconvolve_mixture
Quantify a mixture spectrum using uploaded data.

## Workflow for Reaction Analysis

1. **Determine SMILES**: use your chemistry knowledge to get canonical SMILES for each compound
2. **Predict**: predict_products(reactants="SMILES1 . SMILES2", reagents="catalyst_SMILES")
3. **Lookup spectra**: load_references_for_smiles(smiles_list=<all_species_for_nmr_lookup from step 2>)

## Important
- Join reactant SMILES with " . " (space-dot-space)
- Catalysts/reagents (e.g. FeBr3, H2SO4) go in the reagents parameter, not reactants
- Report ONLY tool results for products and spectra — never invent these"""


# Model configurations: display_name -> (api_url, model_id)
LLM_MODEL_CONFIG = {
    "Llama 3.1 8B (Cerebras)": (
        "https://router.huggingface.co/cerebras/v1",
        "llama3.1-8b"
    ),
    "Llama 3.3 70B (Cerebras)": (
        "https://router.huggingface.co/cerebras/v1",
        "llama-3.3-70b"
    ),
    "DeepSeek V3": (
        "https://router.huggingface.co/sambanova/v1",
        "DeepSeek-V3-0324"
    ),
    "Qwen 2.5 72B": (
        "https://router.huggingface.co/hyperbolic/v1",
        "Qwen/Qwen2.5-72B-Instruct"
    ),
}

LLM_MODELS = list(LLM_MODEL_CONFIG.keys())


@dataclass
class AgentState:
    """State maintained across agent turns."""
    messages: List[Dict[str, Any]] = field(default_factory=list)
    mixture_ppm: Optional[List[float]] = None
    mixture_intensity: Optional[List[float]] = None
    cached_references: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    tool_calls_log: List[Dict[str, Any]] = field(default_factory=list)

    def set_mixture_data(self, ppm: List[float], intensity: List[float]):
        """Set mixture spectrum data from uploaded CSV."""
        self.mixture_ppm = ppm
        self.mixture_intensity = intensity

    def has_mixture_data(self) -> bool:
        """Check if mixture data is available."""
        return self.mixture_ppm is not None and len(self.mixture_ppm) > 0

    def clear(self):
        """Clear all state for a new conversation."""
        self.messages = []
        self.mixture_ppm = None
        self.mixture_intensity = None
        self.cached_references = {}
        self.tool_calls_log = []
    
    def export_session(self) -> Dict[str, Any]:
        """Export the full session for saving/analysis."""
        return {
            "messages": self.messages,
            "tool_calls": self.tool_calls_log,
            "mixture_data": {
                "has_data": self.has_mixture_data(),
                "num_points": len(self.mixture_ppm) if self.mixture_ppm else 0,
            },
            "cached_references": list(self.cached_references.keys()),
        }


class ChemistryAgent:
    """
    LLM Agent for NMR chemistry analysis.

    Uses OpenAI-compatible API with tool calling to autonomously
    execute chemistry analysis workflows.
    """

    def __init__(self, model_name: str = "DeepSeek V3"):
        self.model_name = model_name
        config = LLM_MODEL_CONFIG.get(model_name, LLM_MODEL_CONFIG["DeepSeek V3"])
        self.api_url, self.model_id = config
        self.state = AgentState()

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.api_url,
            api_key=os.environ.get("HF_TOKEN", ""),
        )

    def set_model(self, model_name: str):
        """Change the LLM model."""
        if model_name in LLM_MODEL_CONFIG:
            self.model_name = model_name
            config = LLM_MODEL_CONFIG[model_name]
            self.api_url, self.model_id = config
            self.client = OpenAI(
                base_url=self.api_url,
                api_key=os.environ.get("HF_TOKEN", ""),
            )

    def set_mixture_data(self, ppm: List[float], intensity: List[float]):
        """Set mixture spectrum data."""
        self.state.set_mixture_data(ppm, intensity)

    def clear_state(self):
        """Clear conversation state."""
        self.state.clear()
    
    def export_session(self) -> Dict[str, Any]:
        """Export the full session including model info and reasoning traces."""
        from datetime import datetime
        return {
            "timestamp": datetime.now().isoformat(),
            "model": {
                "name": self.model_name,
                "id": self.model_id,
                "api_url": self.api_url,
            },
            "conversation": self.state.messages,
            "tool_calls": self.state.tool_calls_log,
            "mixture_data": {
                "has_data": self.state.has_mixture_data(),
                "num_points": len(self.state.mixture_ppm) if self.state.mixture_ppm else 0,
            },
            "cached_references": list(self.state.cached_references.keys()),
        }

    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result."""
        try:
            if tool_name == "resolve_chemical_name":
                name = arguments.get("name", "")
                smiles_list = resolve_names_to_smiles(name)
                if smiles_list:
                    return {"smiles": smiles_list[0], "all_matches": smiles_list}
                return {"error": f"Could not resolve chemical name: {name}"}

            elif tool_name == "predict_products":
                reactants = arguments.get("reactants", "")
                reagents = arguments.get("reagents", "")
                n_best = arguments.get("n_best", 5)

                predictions = propose_products(reactants, reagents=reagents, n_best=n_best)
                products = [s for s, _ in predictions]
                unique = get_unique_components(products)
                
                # Extract reactant SMILES for convenience
                reactant_smiles = [s.strip() for s in reactants.split(".") if s.strip()]
                all_species = reactant_smiles + unique

                return {
                    "predictions": [{"smiles": s, "score": float(sc)} for s, sc in predictions],
                    "unique_products": unique,
                    "reactant_smiles": reactant_smiles,
                    "all_species_for_nmr_lookup": all_species,
                    "hint": "Use load_references_for_smiles with all_species_for_nmr_lookup to get NMR spectra for all compounds",
                }

            elif tool_name == "lookup_nmr_reference":
                smiles = arguments.get("smiles", "")
                ref = get_reference_by_smiles(smiles)
                if ref:
                    ppm = ref.get("ppm", [])
                    intensity = ref.get("intensity", [])
                    name = ref.get("name", smiles)
                    
                    result = {
                        "name": name,
                        "smiles": ref.get("smiles", smiles),
                        "ppm": ppm,
                        "intensity": intensity,
                    }
                    # Cache the reference
                    self.state.cached_references[smiles] = result
                    
                    # Generate spectrum plot if we have data
                    if ppm:
                        img_path = plot_spectrum(ppm, intensity, name, style="sticks")
                        result["spectrum_image"] = img_path
                    
                    return result
                return {"error": f"No NMR reference found for SMILES: {smiles}"}

            elif tool_name == "load_references_for_smiles":
                smiles_list = arguments.get("smiles_list", [])
                found = []
                missing = []
                spectrum_images = []

                for smiles in smiles_list:
                    ref = get_reference_by_smiles(smiles)
                    if ref:
                        ppm = ref.get("ppm", [])
                        intensity = ref.get("intensity", [])
                        name = ref.get("name", smiles)
                        
                        ref_data = {
                            "name": name,
                            "smiles": ref.get("smiles", smiles),
                            "ppm": ppm,
                            "intensity": intensity,
                        }
                        found.append(ref_data)
                        self.state.cached_references[smiles] = ref_data
                        
                        # Generate spectrum plot if we have data
                        if ppm:
                            img_path = plot_spectrum(ppm, intensity, name, style="sticks")
                            spectrum_images.append(img_path)
                    else:
                        missing.append(smiles)

                return {
                    "found": len(found),
                    "missing": missing,
                    "references": [{"name": r["name"], "smiles": r["smiles"], "num_peaks": len(r["ppm"])} for r in found],
                    "spectrum_images": spectrum_images,
                }

            elif tool_name == "deconvolve_mixture":
                if not self.state.has_mixture_data():
                    return {"error": "No mixture spectrum uploaded. Please upload a mixture CSV file first."}

                reference_smiles = arguments.get("reference_smiles", [])

                # Build references list
                refs = []
                missing = []
                for smiles in reference_smiles:
                    if smiles in self.state.cached_references:
                        refs.append(self.state.cached_references[smiles])
                    else:
                        # Try to load it
                        ref = get_reference_by_smiles(smiles)
                        if ref:
                            ref_data = {
                                "name": ref.get("name", ""),
                                "smiles": ref.get("smiles", smiles),
                                "ppm": ref.get("ppm", []),
                                "intensity": ref.get("intensity", []),
                            }
                            refs.append(ref_data)
                            self.state.cached_references[smiles] = ref_data
                        else:
                            missing.append(smiles)

                if not refs:
                    return {"error": f"No reference spectra found for any compounds. Missing: {missing}"}

                if missing:
                    # Continue with available refs but note the missing ones
                    pass

                # Use magnetstein if available, otherwise fall back to masserstein
                if _USE_MAGNETSTEIN:
                    result = magnetstein_quantify(
                        mixture_ppm=self.state.mixture_ppm,
                        mixture_intensity=self.state.mixture_intensity,
                        library=refs,
                    )
                    concentrations = result.get("concentrations", {})
                    backend = "magnetstein"
                else:
                    result = deconvolve_spectra(
                        mixture_ppm=self.state.mixture_ppm,
                        mixture_intensity=self.state.mixture_intensity,
                        refs=refs,
                        quiet=True,
                    )
                    concentrations = result.get("concentrations", {})
                    backend = "masserstein"

                return {
                    "concentrations": concentrations,
                    "missing_references": missing,
                    "num_components_used": len(refs),
                    "backend": backend,
                }

            else:
                return {"error": f"Unknown tool: {tool_name}"}

        except Exception as e:
            return {"error": f"Tool execution failed: {str(e)}"}

    def run(self, user_message: str) -> Generator[Tuple[str, List[Dict]], None, None]:
        """
        Run the agent with a user message.

        Yields (partial_response, tool_calls) tuples as the agent works.
        Final yield contains the complete response.
        """
        # Add user message to history
        self.state.messages.append({"role": "user", "content": user_message})

        # Build messages with system prompt
        system_content = SYSTEM_PROMPT
        
        # Inject mixture context if data is available
        if self.state.has_mixture_data():
            ppm_min = min(self.state.mixture_ppm)
            ppm_max = max(self.state.mixture_ppm)
            num_points = len(self.state.mixture_ppm)
            system_content += f"\n\n[CONTEXT: User has uploaded a mixture NMR spectrum with {num_points} data points covering {ppm_min:.2f}-{ppm_max:.2f} ppm. You can use deconvolve_mixture to analyze it.]"
        
        messages = [{"role": "system", "content": system_content}] + self.state.messages

        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            try:
                # Call LLM with tools
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    tools=CHEMISTRY_TOOLS,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=2000,
                )

                choice = response.choices[0]
                message = choice.message

                # Check if we have tool calls
                if message.tool_calls:
                    # Add assistant message with tool calls to history
                    tool_call_msg = {
                        "role": "assistant",
                        "content": message.content or "",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            }
                            for tc in message.tool_calls
                        ]
                    }
                    messages.append(tool_call_msg)

                    # Execute each tool call
                    tool_results = []
                    for tool_call in message.tool_calls:
                        tool_name = tool_call.function.name
                        try:
                            arguments = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError:
                            arguments = {}

                        # Execute the tool
                        result = self._execute_tool(tool_name, arguments)

                        # Log the tool call
                        log_entry = {
                            "tool": tool_name,
                            "arguments": arguments,
                            "result": result,
                        }
                        self.state.tool_calls_log.append(log_entry)
                        tool_results.append(log_entry)

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, default=str),
                        })

                    # Yield intermediate state
                    yield (message.content or f"Executing {len(tool_results)} tool(s)...", tool_results)

                else:
                    # No tool calls - final response
                    final_response = message.content or "Analysis complete."
                    self.state.messages.append({"role": "assistant", "content": final_response})
                    yield (final_response, [])
                    return

            except Exception as e:
                error_msg = f"Error communicating with LLM: {str(e)}"
                self.state.messages.append({"role": "assistant", "content": error_msg})
                yield (error_msg, [])
                return

        # Max iterations reached
        final_msg = "Reached maximum iterations. Here's what I found so far."
        self.state.messages.append({"role": "assistant", "content": final_msg})
        yield (final_msg, [])

    def run_sync(self, user_message: str) -> Tuple[str, List[Dict]]:
        """
        Run the agent synchronously and return final result.

        Returns (final_response, all_tool_calls)
        """
        all_tool_calls = []
        final_response = ""

        for response, tool_calls in self.run(user_message):
            final_response = response
            all_tool_calls.extend(tool_calls)

        return final_response, all_tool_calls


def create_agent(model_name: str = "DeepSeek V3") -> ChemistryAgent:
    """Factory function to create an agent."""
    return ChemistryAgent(model_name=model_name)
