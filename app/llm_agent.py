"""
LLM Agent for NMR Chemistry Analysis
Uses DeepSeek/HuggingFace API to process natural language requests and orchestrate chemistry tools
"""
import os
import json
import re
import requests
import warnings
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

# Suppress HuggingFace tokenizers parallelism warning
warnings.filterwarnings("ignore", message=".*tokenizers.*parallelism.*")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .agent import normalize_smiles_list, step_propose, load_refs_for_species
from .tools_reactiont5 import get_unique_components
from .tools_asics import asics_quantify
from .tools_magnetstein import quantify_single as magnet_quant_single
from .tools_nmrbank import get_reference_by_smiles

@dataclass
class AnalysisTask:
    """Represents a parsed chemistry analysis task"""
    reactants: List[str]
    reagents: str
    analysis_type: str  # "single", "timeseries", "structure_prediction"
    backend_preference: str  # "asics", "magnetstein", "auto"
    additional_context: str

class ChemistryLLMAgent:
    def __init__(self, model_name: str = "deepseek-ai/DeepSeek-V3:together"):
        self.model_name = model_name
        self.api_url = "https://router.huggingface.co/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}",
        }
        
    def _query_llm(self, messages: List[Dict[str, str]]) -> str:
        """Query the LLM with messages and return response"""
        try:
            payload = {
                "messages": messages,
                "model": self.model_name,
                "temperature": 0.1,
                "max_tokens": 1000
            }
            response = requests.post(self.api_url, headers=self.headers, json=payload)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"LLM Error: {str(e)}"

    def parse_chemistry_request(self, user_input: str) -> AnalysisTask:
        """Parse natural language input into structured analysis task"""
        
        system_prompt = """You are a chemistry analysis expert. Parse the user's request and extract:
1. Chemical reactants (names or SMILES)
2. Reagents/conditions 
3. Analysis type (single sample quantification, time-series analysis, or structure prediction)
4. Backend preference (ASICS for standard analysis, Magnetstein for robust analysis, or auto)

Respond with JSON in this exact format:
{
    "reactants": ["compound1", "compound2"],
    "reagents": "conditions or reagents",
    "analysis_type": "single|timeseries|structure_prediction", 
    "backend_preference": "asics|magnetstein|auto",
    "additional_context": "any other relevant details"
}

Common chemical names you should recognize: anisole, benzene, toluene, phenol, bromobenzene, p-bromoanisole, o-bromoanisole, m-bromoanisole, bromine (Br2), iron(III) bromide (FeBr3).
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        response = self._query_llm(messages)
        
        try:
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return AnalysisTask(
                    reactants=parsed.get("reactants", []),
                    reagents=parsed.get("reagents", ""),
                    analysis_type=parsed.get("analysis_type", "single"),
                    backend_preference=parsed.get("backend_preference", "auto"),
                    additional_context=parsed.get("additional_context", "")
                )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Failed to parse LLM response: {e}")
        
        # Fallback parsing
        return self._fallback_parse(user_input)
    
    def _fallback_parse(self, user_input: str) -> AnalysisTask:
        """Fallback parsing using simple heuristics"""
        # Simple keyword extraction
        reactants = []
        reagents = ""
        analysis_type = "single"
        backend_preference = "auto"
        
        # Look for common chemical names
        common_chemicals = ["anisole", "benzene", "toluene", "phenol", "bromobenzene", 
                          "p-bromoanisole", "o-bromoanisole", "m-bromoanisole", 
                          "bromine", "br2", "febr3", "iron bromide"]
        
        text_lower = user_input.lower()
        for chem in common_chemicals:
            if chem in text_lower:
                reactants.append(chem)
        
        # Detect analysis type
        if "time" in text_lower or "series" in text_lower or "kinetic" in text_lower:
            analysis_type = "timeseries"
        elif "structure" in text_lower or "predict" in text_lower:
            analysis_type = "structure_prediction"
        
        # Detect backend preference
        if "magnetstein" in text_lower or "robust" in text_lower:
            backend_preference = "magnetstein"
        elif "asics" in text_lower:
            backend_preference = "asics"
            
        return AnalysisTask(
            reactants=reactants,
            reagents=reagents,
            analysis_type=analysis_type,
            backend_preference=backend_preference,
            additional_context=user_input
        )

    def execute_analysis(self, task: AnalysisTask, mixture_data: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
        """Execute the chemistry analysis based on the parsed task"""
        
        results = {
            "task": task,
            "steps": [],
            "final_results": {},
            "narrative": ""
        }
        
        try:
            # Step 1: Normalize reactant SMILES
            if task.reactants:
                normalized_reactants = normalize_smiles_list(", ".join(task.reactants))
                results["steps"].append({
                    "step": "normalize_reactants",
                    "input": task.reactants,
                    "output": normalized_reactants,
                    "status": "success"
                })
            else:
                results["steps"].append({
                    "step": "normalize_reactants",
                    "status": "skipped",
                    "reason": "No reactants specified"
                })
                normalized_reactants = []

            # Step 2: Propose products if reactants are available
            products = []
            if normalized_reactants:
                products = step_propose(normalized_reactants, task.reagents, topk=5)
                results["steps"].append({
                    "step": "propose_products",
                    "input": {"reactants": normalized_reactants, "reagents": task.reagents},
                    "output": products,
                    "status": "success"
                })
            else:
                results["steps"].append({
                    "step": "propose_products",
                    "status": "skipped",
                    "reason": "No valid reactants found"
                })

            # Step 3: Extract unique components and load reference spectra
            unique_components = get_unique_components(products) if products else []
            all_species = normalized_reactants + unique_components
            refs = []
            if all_species:
                refs = load_refs_for_species(all_species)
                results["steps"].append({
                    "step": "load_references",
                    "input": all_species,
                    "output": [{"name": r["name"], "smiles": r["smiles"]} for r in refs],
                    "status": "success"
                })

            # Step 4: Quantification (if mixture data provided)
            if mixture_data and refs and task.analysis_type == "single":
                backend = task.backend_preference if task.backend_preference != "auto" else "asics"
                
                if backend == "magnetstein":
                    quant_result = magnet_quant_single(
                        mixture_data["ppm"], 
                        mixture_data["intensity"], 
                        refs
                    )
                elif backend == "deconvolve":
                    # Use external Masserstein+Gurobi deconvolution script (moved into app/)
                    import os, json, tempfile, subprocess
                    script_path = "/Users/luciavinalopez/LLMHackathon/app/tool_deconvolve_nmr.py"
                    license_path = "/Users/luciavinalopez/LLMHackathon/magnetstein/gurobi.lic"
                    with tempfile.TemporaryDirectory() as td:
                        # Write mixture CSV
                        mix_path = os.path.join(td, "mixture.csv")
                        with open(mix_path, "w") as f:
                            for x, y in zip(mixture_data["ppm"], mixture_data["intensity"]):
                                f.write(f"{x},{y}\n")
                        # Write component CSVs
                        comp_paths = []
                        names = []
                        for i, r in enumerate(refs):
                            cpath = os.path.join(td, f"comp_{i}.csv")
                            with open(cpath, "w") as f:
                                for x, y in zip(r["ppm"], r["intensity"]):
                                    f.write(f"{x},{y}\n")
                            comp_paths.append(cpath)
                            names.append(r.get("name", f"comp{i}"))
                        cmd = [
                            os.environ.get("PYTHON", "python"),
                            script_path,
                            mix_path,
                            *comp_paths,
                            "--names",
                            *names,
                            "--json",
                            "--quiet",
                        ]
                        if os.path.exists(license_path):
                            cmd += ["--license-file", license_path]
                        proc = subprocess.run(cmd, capture_output=True, text=True)
                        stdout = proc.stdout or ""
                        json_line = ""
                        for ln in stdout.splitlines()[::-1]:
                            if ln.startswith("JSON:"):
                                json_line = ln[len("JSON:"):].strip()
                                break
                        if not json_line:
                            start = stdout.rfind("{")
                            end = stdout.rfind("}")
                            if start != -1 and end != -1 and end > start:
                                json_line = stdout[start:end+1]
                        payload = json.loads(json_line) if json_line else {"proportions": {}}
                        conc = payload.get("proportions", {})
                        if not conc:
                            import re
                            try:
                                grab = False
                                est = {}
                                for ln in stdout.splitlines():
                                    if ln.strip().lower().startswith("estimated proportions"):
                                        grab = True
                                        continue
                                    if grab:
                                        m = re.match(r"\s*(.+?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$", ln)
                                        if m:
                                            est[m.group(1).strip()] = float(m.group(2))
                                if est:
                                    conc = est
                            except Exception:
                                pass
                        quant_result = {"concentrations": conc, "raw": payload}
                else:
                    quant_result = asics_quantify(
                        crude_ppm=mixture_data["ppm"],
                        crude_intensity=mixture_data["intensity"],
                        refs=[{"name": r["name"], "ppm": r["ppm"], "intensity": r["intensity"]} for r in refs],
                        max_shift=0.02,
                        quant_method="FWER"
                    )
                
                results["steps"].append({
                    "step": "quantification",
                    "input": {"backend": backend, "num_refs": len(refs)},
                    "output": quant_result,
                    "status": "success"
                })
                results["final_results"]["quantification"] = quant_result

            # Generate narrative
            results["narrative"] = self._generate_narrative(results)
            
        except Exception as e:
            results["steps"].append({
                "step": "error",
                "status": "failed",
                "error": str(e)
            })
            results["narrative"] = f"Analysis failed with error: {str(e)}"
        
        return results

    def _generate_narrative(self, results: Dict[str, Any]) -> str:
        """Generate a natural language narrative of the analysis"""
        
        task = results["task"]
        steps = results["steps"]
        
        # Create summary for LLM
        summary_parts = []
        for step in steps:
            if step["status"] == "success":
                if step["step"] == "normalize_reactants":
                    summary_parts.append(f"Identified reactants: {step['output']}")
                elif step["step"] == "propose_products":
                    summary_parts.append(f"Proposed products: {step['output']}")
                elif step["step"] == "load_references":
                    refs_found = [r["name"] for r in step["output"]]
                    summary_parts.append(f"Found reference spectra for: {refs_found}")
                elif step["step"] == "quantification":
                    backend = step["input"]["backend"]
                    summary_parts.append(f"Performed quantification using {backend}")
        
        system_prompt = """You are a chemistry expert. Generate a clear, professional narrative summary of the NMR analysis results. 
Be specific about what was found and what methods were used. Keep it concise but informative."""
        
        user_prompt = f"""
Original request: {task.additional_context}
Analysis steps completed: {'; '.join(summary_parts)}

Please provide a narrative summary of this NMR chemistry analysis.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        narrative = self._query_llm(messages)
        return narrative if not narrative.startswith("LLM Error") else "Analysis completed successfully."

# Global agent instance
_agent = None

def get_agent() -> ChemistryLLMAgent:
    """Get or create the global agent instance"""
    global _agent
    if _agent is None:
        _agent = ChemistryLLMAgent()
    return _agent

def process_natural_language_request(user_input: str, mixture_data: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
    """Main entry point for processing natural language chemistry requests"""
    agent = get_agent()
    task = agent.parse_chemistry_request(user_input)
    return agent.execute_analysis(task, mixture_data)
