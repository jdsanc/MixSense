"""
OpenAI-compatible tool definitions for the Chemistry Agent.

These schemas define the tools available to the LLM agent for
NMR chemistry analysis tasks.
"""

CHEMISTRY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "resolve_chemical_name",
            "description": "Convert a chemical name (e.g., 'anisole', 'benzene', 'alpha-pinene') to its canonical SMILES representation. Use this tool first when the user mentions chemicals by name rather than SMILES.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The chemical name to resolve (e.g., 'anisole', 'bromine', 'FeBr3')"
                    }
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_products",
            "description": "Predict reaction products using the ReactionT5 model. Takes reactant SMILES (separated by ' . ') and optional reagents/conditions. Returns predicted product SMILES with confidence scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reactants": {
                        "type": "string",
                        "description": "Reactant SMILES separated by ' . ' (e.g., 'COc1ccccc1 . BrBr' for anisole + bromine)"
                    },
                    "reagents": {
                        "type": "string",
                        "description": "Optional reagents or reaction conditions (e.g., 'FeBr3' for Lewis acid catalyst)",
                        "default": ""
                    },
                    "n_best": {
                        "type": "integer",
                        "description": "Number of top predictions to return",
                        "default": 5
                    }
                },
                "required": ["reactants"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_nmr_reference",
            "description": "Look up the NMR reference spectrum for a single compound by its SMILES. Returns chemical shifts (ppm) and intensities. Use load_references_for_smiles for multiple compounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {
                        "type": "string",
                        "description": "SMILES string of the compound to look up"
                    }
                },
                "required": ["smiles"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "load_references_for_smiles",
            "description": "Load NMR reference spectra for multiple SMILES strings at once. More efficient than calling lookup_nmr_reference multiple times. Returns found references and lists any missing compounds.",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles_list": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of SMILES strings to look up"
                    }
                },
                "required": ["smiles_list"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deconvolve_mixture",
            "description": "Deconvolve/quantify an NMR mixture spectrum into component contributions. Uses the mixture spectrum uploaded by the user and the reference spectra previously loaded. Returns relative concentrations of each component.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reference_smiles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of SMILES for compounds expected in the mixture. References will be loaded automatically."
                    }
                },
                "required": ["reference_smiles"]
            }
        }
    }
]


# Map tool names to their parameter schemas for validation
TOOL_SCHEMAS = {tool["function"]["name"]: tool["function"]["parameters"] for tool in CHEMISTRY_TOOLS}
