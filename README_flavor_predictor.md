# Flavor Prediction Script

A Python script that uses the FART_Augmented model from Hugging Face to predict molecular flavors from SMILES strings.

## Overview

The Flavor Analysis and Recognition Transformer (FART) is a state-of-the-art machine learning model that predicts molecular taste from chemical structures encoded as SMILES. The model can classify molecules into five categories:

- **Sweet**: Sweet-tasting compounds
- **Bitter**: Bitter-tasting compounds  
- **Sour**: Sour-tasting compounds
- **Umami**: Umami-tasting compounds
- **Undefined**: Tasteless or undefined compounds

## Installation

Install the required dependencies:

```bash
pip install torch transformers numpy
```

Or use the requirements file:

```bash
pip install -r requirements_flavor.txt
```

## Usage

### Command Line Interface

#### Single SMILES string:
```bash
python flavor_predictor.py "CCO"
```

#### Multiple SMILES strings:
```bash
python flavor_predictor.py "CCO" "CCN" "CC(=O)O"
```

#### From file:
```bash
python flavor_predictor.py --file example_smiles.txt
```

#### Verbose output (shows all probabilities):
```bash
python flavor_predictor.py "CCO" --verbose
```

#### Help:
```bash
python flavor_predictor.py --help
```

### Programmatic Usage

```python
from flavor_predictor import FlavorPredictor

# Initialize predictor
predictor = FlavorPredictor()

# Predict single SMILES
result = predictor.predict_single("CCO")
print(f"Predicted flavor: {result['predicted_flavor']}")

# Predict multiple SMILES
smiles_list = ["CCO", "CCN", "CC(=O)O"]
results = predictor.predict_flavor(smiles_list)

for result in results:
    print(f"SMILES: {result['smiles']} -> {result['predicted_flavor']}")
```

## Example Results

Here are some example predictions:

| SMILES | Molecule | Predicted Flavor | Confidence |
|--------|----------|------------------|------------|
| `CCO` | Ethanol | Sweet | 0.464 |
| `CCN` | Ethylamine | Undefined | 0.987 |
| `CC(=O)O` | Acetic acid | Sour | 0.644 |
| `C1=CC=C(C=C1)O` | Phenol | Sweet | 0.583 |
| `C1=CC=C(C=C1)N` | Aniline | Bitter | 0.466 |

## Model Information

- **Model**: [FartLabs/FART_Augmented](https://huggingface.co/FartLabs/FART_Augmented)
- **Architecture**: RoBERTa-based transformer
- **Training Data**: 15,025 curated molecular tastants
- **Accuracy**: >91% on test set
- **Input**: SMILES strings
- **Output**: Flavor classification with confidence scores

## Files

- `flavor_predictor.py` - Main prediction script
- `example_usage.py` - Example programmatic usage
- `example_smiles.txt` - Sample SMILES strings for testing
- `requirements_flavor.txt` - Python dependencies
- `README_flavor_predictor.md` - This documentation

## Features

- ✅ Command-line interface with multiple input options
- ✅ Programmatic API for integration into other projects
- ✅ Batch processing of multiple SMILES strings
- ✅ File input support
- ✅ Detailed probability output
- ✅ Error handling and input validation
- ✅ GPU support (automatic detection)
- ✅ Comprehensive documentation

## Notes

- The model is based on the ChemBERTa pre-trained transformer
- First-time usage will download the model (~334MB) from Hugging Face
- The model performs multi-class classification with softmax probabilities
- Results include confidence scores and all class probabilities
- The model works best with small molecules (typical molecular weight ~374 Da)

## References

- [FART_Augmented Model on Hugging Face](https://huggingface.co/FartLabs/FART_Augmented)
- [FART Repository](https://github.com/fart-lab/fart) (submodule in this project)
- Original paper: Flavor Analysis and Recognition Transformer (FART) for molecular taste prediction
