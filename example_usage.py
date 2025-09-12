#!/usr/bin/env python3
"""
Example usage of the FlavorPredictor class

This script demonstrates how to use the FlavorPredictor class programmatically
to predict flavors from SMILES strings.
"""

from flavor_predictor import FlavorPredictor

def main():
    """Example usage of the FlavorPredictor class."""
    
    # Initialize the predictor
    print("Initializing FlavorPredictor...")
    predictor = FlavorPredictor()
    
    # Example SMILES strings representing different types of molecules
    example_smiles = [
        "CCO",                    # Ethanol (alcohol)
        "CCN",                    # Ethylamine (amine)
        "CC(=O)O",                # Acetic acid (carboxylic acid)
        "C1=CC=C(C=C1)O",         # Phenol (aromatic alcohol)
        "CC(C)CO",                # Isobutanol (branched alcohol)
        "C1=CC=CC=C1",            # Benzene (aromatic hydrocarbon)
        "CC(=O)OC1=CC=CC=C1C(=O)O", # Aspirin (ester)
        "C1=CC=C(C=C1)C(=O)O",    # Benzoic acid (aromatic carboxylic acid)
        "CC(C)(C)CO",             # Neopentyl alcohol
        "C1=CC=C(C=C1)N"          # Aniline (aromatic amine)
    ]
    
    print(f"\nPredicting flavors for {len(example_smiles)} example molecules...")
    
    # Make predictions
    results = predictor.predict_flavor(example_smiles)
    
    # Display results
    print("\n" + "="*100)
    print("EXAMPLE FLAVOR PREDICTIONS")
    print("="*100)
    
    for i, result in enumerate(results, 1):
        smiles = result['smiles']
        flavor = result['predicted_flavor']
        confidence = result['confidence']
        
        print(f"{i:2d}. {smiles:30s} -> {flavor:12s} (confidence: {confidence:.3f})")
    
    print("\n" + "="*100)
    
    # Show detailed results for a few examples
    print("\nDetailed results for first 3 molecules:")
    for i, result in enumerate(results[:3], 1):
        print(f"\n{i}. SMILES: {result['smiles']}")
        print(f"   Predicted Flavor: {result['predicted_flavor']}")
        print(f"   Confidence: {result['confidence']:.3f}")
        print("   All Probabilities:")
        for flavor, prob in sorted(result['all_probabilities'].items(), 
                                 key=lambda x: x[1], reverse=True):
            print(f"     {flavor:12s}: {prob:.3f}")

if __name__ == "__main__":
    main()
