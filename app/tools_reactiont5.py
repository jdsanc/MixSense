# app/tools_reactiont5.py
from typing import List, Tuple
import requests
import os
import sys
from pathlib import Path
from rdkit import Chem

# Add the reactants_to_products module to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from reactants_to_products.reactants_to_products import (
        predict_products as local_predict_products,
        add_water_for_new_esters,
        unique_components_from_balanced
    )
    LOCAL_MODEL_AVAILABLE = True
    print("Local ReactionT5 model available")
except ImportError as e:
    LOCAL_MODEL_AVAILABLE = False
    print(f"Local model not available, will use API: {e}")

API_URL = "https://api-inference.huggingface.co/models/sagawa/ReactionT5v2-forward"
headers = {
    "Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}",
}

def _query_hf_api(payload):
    """Query Hugging Face Inference API"""
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None

def _valid_smiles(s: str) -> bool:
    try:
        return Chem.MolFromSmiles(s) is not None
    except Exception:
        return False

def propose_products(reactants: str, reagents: str = "", beams=10, n_best=5, max_new_tokens=64) -> List[Tuple[str, float]]:
    """
    Returns top-n product SMILES with scores using local model first, with API fallback.
    Enhanced with water balancing for ester formation reactions.
    Input format: 'REACTANT_SMILES . REACTANT_SMILES' for reactants
    """
    
    # Try local model first
    if LOCAL_MODEL_AVAILABLE:
        try:
            print("Using local ReactionT5 model...")
            
            # Use the enhanced local model with water balancing
            raw_predictions = local_predict_products(
                reactants=reactants,
                reagents=reagents,
                beams=beams,
                n_best=n_best,
                max_new_tokens=max_new_tokens
            )
            
            # Apply water balancing for each prediction
            balanced_predictions = [
                add_water_for_new_esters(reactants, pred) 
                for pred in raw_predictions
            ]
            
            # Remove duplicates after balancing
            balanced_predictions = list(dict.fromkeys(balanced_predictions))
            
            # Convert to expected format with scores
            results = [(pred, 1.0) for pred in balanced_predictions[:n_best]]
            
            if results:
                print(f"Local model generated {len(results)} predictions with water balancing")
                return results
            else:
                print("Local model returned no valid results, falling back to API...")
                
        except Exception as e:
            print(f"Local model failed: {e}, falling back to API...")
    
    # Fallback to API
    return _api_propose_products(reactants, reagents, beams, n_best, max_new_tokens)


def _api_propose_products(reactants: str, reagents: str, beams: int, n_best: int, max_new_tokens: int) -> List[Tuple[str, float]]:
    """
    Original API-based product prediction (now as fallback)
    """
    print("Using HuggingFace API...")
    prompt = f"{reactants} > {reagents} >"
    
    # Query Hugging Face API
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_new_tokens,
            "num_beams": beams,
            "num_return_sequences": min(n_best, beams),
            "do_sample": False,
            "early_stopping": True
        }
    }
    
    response = _query_hf_api(payload)
    if not response:
        print("Failed to get response from Hugging Face API")
        return []
    
    # Handle different response formats
    if isinstance(response, list) and len(response) > 0:
        # Standard format: [{"generated_text": "..."}, ...]
        texts = [item.get("generated_text", "") for item in response]
    elif isinstance(response, dict) and "generated_text" in response:
        # Single response format: {"generated_text": "..."}
        texts = [response["generated_text"]]
    else:
        print(f"Unexpected response format: {response}")
        return []
    
    # Process results
    results = []
    for t in texts:
        s = t.split()[0].strip()
        if _valid_smiles(s):
            results.append((s, 1.0))  # placeholder score
    
    # Deduplicate preserving order
    seen, dedup = set(), []
    for s, sc in results:
        if s not in seen:
            seen.add(s)
            dedup.append((s, sc))
    
    return dedup[:n_best]


def get_unique_components(predictions: List[str]) -> List[str]:
    """
    Extract unique components from all product predictions.
    Uses the enhanced function from reactants_to_products if available.
    """
    if LOCAL_MODEL_AVAILABLE:
        return unique_components_from_balanced(predictions)
    else:
        # Fallback implementation
        seen = set()
        components = []
        for prediction in predictions:
            for component in prediction.split('.'):
                component = component.strip()
                if component and component not in seen and _valid_smiles(component):
                    seen.add(component)
                    components.append(component)
        return components
