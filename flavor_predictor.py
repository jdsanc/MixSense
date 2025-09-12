#!/usr/bin/env python3
"""
Flavor Prediction Script using FART_Augmented Model

This script uses the FART_Augmented model from Hugging Face to predict molecular flavors
from SMILES strings. The model can classify molecules into five categories:
- sweet
- bitter  
- sour
- umami
- undefined (tasteless)

Usage:
    python flavor_predictor.py "CCO"  # Single SMILES
    python flavor_predictor.py "CCO" "CCN" "CC(=O)O"  # Multiple SMILES
    python flavor_predictor.py --file smiles.txt  # From file
"""

import argparse
import sys
import torch
from typing import Union, List, Dict, Any
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class FlavorPredictor:
    """
    A class to predict molecular flavors from SMILES strings using the FART_Augmented model.
    """
    
    def __init__(self, model_name: str = "FartLabs/FART_Augmented"):
        """
        Initialize the flavor predictor with the FART_Augmented model.
        
        Args:
            model_name (str): Name of the Hugging Face model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading FART_Augmented model from {model_name}...")
        self._load_model()
        print(f"Model loaded successfully on {self.device}")
    
    def _load_model(self):
        """Load the model and tokenizer from Hugging Face."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32
            )
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def predict_flavor(self, smiles_input: Union[str, List[str]]) -> List[Dict[str, Any]]:
        """
        Predict flavors for given SMILES strings.
        
        Args:
            smiles_input (Union[str, List[str]]): SMILES string or list of SMILES strings
            
        Returns:
            List[Dict[str, Any]]: List of prediction results with SMILES, predicted flavor, and confidence scores
        """
        # Ensure input is a list
        if isinstance(smiles_input, str):
            smiles_list = [smiles_input]
        else:
            smiles_list = smiles_input
        
        if not smiles_list:
            raise ValueError("No SMILES strings provided")
        
        # Validate SMILES strings
        for smiles in smiles_list:
            if not isinstance(smiles, str) or not smiles.strip():
                raise ValueError(f"Invalid SMILES string: {smiles}")
        
        try:
            # Tokenize the input
            inputs = self.tokenizer(
                smiles_list, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            )
            
            # Move inputs to the same device as model
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get prediction probabilities
            probabilities = torch.softmax(outputs.logits, dim=-1)
            
            # Get predicted class indices
            predictions = torch.argmax(probabilities, dim=-1)
            
            # Get confidence scores
            confidence_scores = torch.max(probabilities, dim=-1)[0]
            
            # Map class indices to flavor labels
            if hasattr(self.model.config, 'id2label') and self.model.config.id2label:
                flavor_labels = self.model.config.id2label
            else:
                # Fallback labels based on the FART model documentation
                flavor_labels = {
                    0: "sweet",
                    1: "bitter", 
                    2: "sour",
                    3: "umami",
                    4: "undefined"
                }
            
            # Compile results
            results = []
            for i, smiles in enumerate(smiles_list):
                predicted_idx = predictions[i].item()
                confidence = confidence_scores[i].item()
                
                # Get all class probabilities for this prediction
                all_probs = probabilities[i].cpu().numpy()
                prob_dict = {
                    flavor_labels.get(j, f"class_{j}"): float(all_probs[j]) 
                    for j in range(len(all_probs))
                }
                
                result = {
                    "smiles": smiles,
                    "predicted_flavor": flavor_labels.get(predicted_idx, f"class_{predicted_idx}"),
                    "confidence": confidence,
                    "all_probabilities": prob_dict
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def predict_single(self, smiles: str) -> Dict[str, Any]:
        """
        Predict flavor for a single SMILES string.
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            Dict[str, Any]: Prediction result
        """
        results = self.predict_flavor(smiles)
        return results[0]


def load_smiles_from_file(filename: str) -> List[str]:
    """
    Load SMILES strings from a text file.
    
    Args:
        filename (str): Path to the file containing SMILES strings
        
    Returns:
        List[str]: List of SMILES strings
    """
    try:
        with open(filename, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        return smiles_list
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filename}")
    except Exception as e:
        raise RuntimeError(f"Error reading file {filename}: {e}")


def print_results(results: List[Dict[str, Any]], verbose: bool = False):
    """
    Print prediction results in a formatted way.
    
    Args:
        results (List[Dict[str, Any]]): List of prediction results
        verbose (bool): Whether to show detailed probability information
    """
    print("\n" + "="*80)
    print("FLAVOR PREDICTION RESULTS")
    print("="*80)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. SMILES: {result['smiles']}")
        print(f"   Predicted Flavor: {result['predicted_flavor']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        
        if verbose:
            print("   All Probabilities:")
            for flavor, prob in sorted(result['all_probabilities'].items(), 
                                     key=lambda x: x[1], reverse=True):
                print(f"     {flavor}: {prob:.3f}")
    
    print("\n" + "="*80)


def main():
    """Main function to handle command line arguments and run predictions."""
    parser = argparse.ArgumentParser(
        description="Predict molecular flavors from SMILES strings using FART_Augmented model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python flavor_predictor.py "CCO"                    # Single SMILES
  python flavor_predictor.py "CCO" "CCN" "CC(=O)O"    # Multiple SMILES
  python flavor_predictor.py --file smiles.txt        # From file
  python flavor_predictor.py "CCO" --verbose          # Detailed output
        """
    )
    
    parser.add_argument(
        "smiles", 
        nargs="*", 
        help="SMILES string(s) to predict flavors for"
    )
    parser.add_argument(
        "--file", "-f", 
        type=str, 
        help="File containing SMILES strings (one per line)"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Show detailed probability information"
    )
    parser.add_argument(
        "--model", "-m", 
        type=str, 
        default="FartLabs/FART_Augmented",
        help="Hugging Face model name (default: FartLabs/FART_Augmented)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.smiles and not args.file:
        parser.error("Must provide either SMILES strings or a file with --file")
    
    try:
        # Initialize predictor
        predictor = FlavorPredictor(model_name=args.model)
        
        # Collect SMILES strings
        smiles_list = []
        
        if args.smiles:
            smiles_list.extend(args.smiles)
        
        if args.file:
            file_smiles = load_smiles_from_file(args.file)
            smiles_list.extend(file_smiles)
        
        if not smiles_list:
            print("No valid SMILES strings found.")
            return
        
        # Make predictions
        print(f"\nPredicting flavors for {len(smiles_list)} SMILES string(s)...")
        results = predictor.predict_flavor(smiles_list)
        
        # Print results
        print_results(results, verbose=args.verbose)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
