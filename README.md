# 🧪 NMR Chemistry Analysis Platform

NMR (Nuclear Magnetic Resonance) chemistry analysis platform. Provides tools for reaction product prediction, NMR reference lookup, and mixture quantification.

## Quick Start

### Install Dependencies
```bash
conda env create -f environment.yml -n mixsense
conda activate mixsense
```

### Run the Gradio UI
```bash
python -m app.gradio_llm_app
# Launches at localhost:7667
```

### Run the MCP Server
```bash
python -m app.mcp_server
# Exposes chemistry tools via Model Context Protocol
```

### Run the CLI Agent
```bash
python -m app.simple_agent --reactants "anisole" "Br2" --reagents "FeBr3"
```

---

## Architecture

### MCP Server (`app/mcp_server.py`)
Exposes chemistry tools via Model Context Protocol:
- `lookup_nmr_reference(smiles)` - Find NMR reference by SMILES
- `resolve_chemical_name(name)` - Convert name to SMILES
- `predict_products(reactants, reagents)` - Predict reaction products
- `deconvolve_mixture(ppm, intensity, refs)` - Quantify mixture components
- `load_references_for_smiles(smiles_list)` - Batch reference lookup

### Core Tools (`app/tools_*.py`)

| Module | Description |
|--------|-------------|
| `tools_nmrbank.py` | NMRBank reference lookup. Lazy-loads CSV into LUT keyed by canonical SMILES. |
| `tools_reactiont5.py` | Product prediction using ReactionT5 model. Local model preferred, falls back to HuggingFace API. |
| `tools_deconvolve.py` | Wrapper for Masserstein+Gurobi deconvolution. |
| `tools_magnetstein.py` | Alternative quantification using magnetstein's optimal transport. |
| `tools_asics.py` | R-based ASICS quantification. |

### LLM Agent (`app/chemistry_agent.py`)
Autonomous agent that chains tools via OpenAI-compatible tool calling:
1. Resolve reactant names → SMILES
2. Predict products
3. Find reference spectra
4. Optionally quantify mixture

### Gradio UI (`app/gradio_llm_app.py`)
Web interface with:
- **Agent Chat** - Autonomous LLM agent with tool calling
- **Manual Deconvolution** - Step-by-step analysis control

### Submodules
- **magnetstein/** - Masserstein optimal transport library
- **NMRBank/** - Reference NMR spectra database (~156k compounds)

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace API token (required for LLM agent) |
| `GRB_LICENSE_FILE` | [Gurobi license](https://www.gurobi.com/academia/academic-program-and-licenses/) file path |
| `NMRBANK_CSV` | Override NMRBank CSV location |
| `NMRBANK_SKIP_LOAD_FOR_TESTS` | Set to `1` to skip CSV load in tests |

---

## Data Formats

**NMR Spectra CSV**: Two columns (ppm, intensity), no header. TSV also supported.

**Reference Library**: List of dicts: `{name, ppm: [...], intensity: [...], smiles}`

---

## NMR Mixture Deconvolution CLI

**Script:** `app/tool_deconvolve_nmr.py`  
**Goal:** Deconvolve a 1D NMR mixture spectrum into contributions from known component spectra using [Masserstein](https://github.com/BDomzal/masserstein) optimal transport and the Gurobi linear programming solver.

### Inputs

**Required:**
- **Mixture spectrum** (`mixture`): CSV/TSV with columns (ppm, intensity)
- **Component spectra** (`components`): One or more CSV/TSV files with the same format

**Optional:**
- `--protons`: Number of protons per component (for intensity normalization)
- `--names`: Custom names for components (for output table and JSON)
- `--mnova`: If files are Mnova TSV exports

### Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `--threads` | Number of BLAS / solver threads | 8 |
| `--time-limit` | Gurobi solver time limit (seconds) | 300 |
| `--method` | LP solve method: `auto`, `primal`, `dual`, `barrier` | dual |
| `--presolve` | Gurobi presolve level (0–2) | 2 |
| `--numeric-focus` | NumericFocus (0–3, higher = more numerical stability) | 1 |
| `--skip-crossover` | Skip crossover when using barrier method | off |
| `--license-file` | Path to `gurobi.lic` license file | env |
| `--json` | Print output also in JSON | off |
| `--quiet` | Reduce solver verbosity | off |

### Example Usage

```bash
python app/tool_deconvolve_nmr.py \
  mixture.csv comp1.csv comp2.csv \
  --protons 16 12 \
  --names "Component 1" "Component 2" \
  --json
```

---

## Dependencies

- Python 3.10+
- rdkit, transformers, torch (for ReactionT5)
- mcp (for MCP server)
- gradio (for web UI)
- Gurobi optimizer (optional, for full deconvolution)

---

## Evaluation


```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MixSense Pipeline                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: "anisole + Br2 with FeBr3"                                      │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────┐                                                │
│  │ Step 1: Name        │  "anisole" → "COc1ccccc1"                      │
│  │ Resolution          │  "Br2"     → "BrBr"                            │
│  └─────────────────────┘  "FeBr3"   → "Br[Fe](Br)Br"                    │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────┐                                                │
│  │ Step 2: Reaction    │  ReactionT5 predicts:                          │
│  │ Prediction          │  → "COc1ccc(Br)cc1" (p-bromoanisole)           │
│  └─────────────────────┘  → "COc1ccccc1Br" (o-bromoanisole)             │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────┐                                                │
│  │ Step 3: Reference   │  Query NMRBank for:                            │
│  │ Lookup              │  - Anisole spectrum                            │
│  └─────────────────────┘  - p-Bromoanisole spectrum                     │
│           │                                                             │
│           ▼                                                             │
│  ┌─────────────────────┐                                                │
│  │ Step 4: Mixture     │  Given experimental spectrum,                  │
│  │ Deconvolution       │  solve: mixture = Σ(cᵢ × refᵢ)                 │
│  └─────────────────────┘                                                │
│           │                                                             │
│           ▼                                                             │
│  OUTPUT: {anisole: 30%, p-bromoanisole: 70%}                            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```


### Step 1: Chemical Name Resolution

**What happens:** Convert human-readable chemical names to machine-readable SMILES (Simplified Molecular Input Line Entry System).

**Tool:** `resolve_names_to_smiles()` in `tools_reactiont5.py`

**Example:**
```
Input:  "anisole, Br2, FeBr3"
Output: ["COc1ccccc1", "BrBr", "Br[Fe](Br)Br"]
```

**How it works:**
1. First tries RDKit to parse as SMILES directly
2. Falls back to a lookup table for common chemicals
3. Can use OPSIN/PubChem for advanced name resolution

**What can go wrong:**
- Unknown chemical names → empty result
- Ambiguous names (e.g., "pinene" could be α or β)
- Misspellings

**Evaluation criteria:**
- Did we get valid SMILES?
- Is the SMILES chemically correct for the name?

---

### Step 2: Reaction Product Prediction

**What happens:** Given reactant SMILES and optional reagent/catalyst, predict likely products using a machine learning model.

**Tool:** `propose_products()` in `tools_reactiont5.py`

**Model:** ReactionT5v2 (transformer model trained on chemical reactions)

**Example:**
```
Input:
  reactants = "COc1ccccc1 . BrBr"   (anisole + bromine)
  reagents  = "Br[Fe](Br)Br"        (FeBr3 catalyst)

Output: [
  ("COc1ccc(Br)cc1", 0.95),   # para-bromoanisole (95% confidence)
  ("COc1ccccc1Br", 0.87),     # ortho-bromoanisole (87% confidence)
]
```

**How it works:**
1. Format: `"REACTANT1 . REACTANT2 > REAGENT >"`
2. ReactionT5 generates product SMILES via beam search
3. Returns top-N predictions with confidence scores

**What can go wrong:**
- Model not trained on this reaction type
- Wrong regiochemistry (ortho vs para)
- Missing byproducts (H₂O, HBr, etc.)
- Hallucinated invalid SMILES

**Evaluation criteria:**
- Is the correct major product in top-1? top-5?
- Are predicted products chemically valid?
- Is regioselectivity correct?

---

### Step 3: NMR Reference Spectrum Lookup

**What happens:** Retrieve reference ¹H NMR spectra from NMRBank database for each compound of interest.

**Tool:** `get_reference_by_smiles()` in `tools_nmrbank.py`

**Database:** NMRBank (~156,000 compounds with experimental NMR data)

**Example:**
```
Input:  "COc1ccccc1" (anisole)

Output: {
  "name": "anisole",
  "smiles": "COc1ccccc1",
  "ppm": [7.28, 6.92, 3.80],
  "intensity": [0.67, 1.0, 1.0],
  "protons": 8
}
```

**How it works:**
1. Canonicalize query SMILES with RDKit
2. Look up in hash table (keyed by canonical SMILES)
3. Parse stored NMR text to extract peaks and integrals

**What can go wrong:**
- Compound not in database (coverage gap)
- Poor quality reference data
- Parsing errors for unusual NMR formats

**Evaluation criteria:**
- Was the compound found?
- Does the spectrum have reasonable peaks?
- Are peak positions chemically sensible?

---

### Step 4: Mixture Deconvolution (Quantification)

**What happens:** Given an experimental mixture spectrum and reference spectra, determine the relative concentration of each component.

**Tools:**
- `quantify_single()` in `tools_magnetstein.py` (optimal transport)
- `deconvolve_spectra()` in `tools_deconvolve.py` (linear programming)

**Mathematical formulation:**
```
minimize:  Wasserstein distance(mixture, Σ cᵢ × refᵢ)
subject to: cᵢ ≥ 0, Σ cᵢ = 1
```

**Example:**
```
Input:
  mixture_spectrum = [(7.28, 0.3), (6.92, 0.4), (3.80, 0.3), (7.38, 0.5), ...]
  references = [anisole_spectrum, bromoanisole_spectrum]

Output: {
  "concentrations": {
    "anisole": 0.32,
    "p-bromoanisole": 0.68
  }
}
```

**How it works:**
1. Convert spectra to probability distributions
2. Solve optimal transport problem
3. Find coefficients that best reconstruct mixture

**What can go wrong:**
- Overlapping peaks → ambiguity
- Missing reference spectrum
- Noise in experimental data
- Non-linear concentration effects

**Evaluation criteria:**
- How close are predicted concentrations to ground truth?
- MAE (Mean Absolute Error)
- Can we detect trace components?

describe baselines, metrics closer etc



### 4.1 Name Resolution Metrics

```python
def evaluate_name_resolution(test_cases):
    """
    test_cases = [
        {"name": "anisole", "expected_smiles": "COc1ccccc1"},
        {"name": "Br2", "expected_smiles": "BrBr"},
        ...
    ]
    """
    correct = 0
    for case in test_cases:
        result = resolve_names_to_smiles(case["name"])
        if result and canonicalize(result[0]) == canonicalize(case["expected_smiles"]):
            correct += 1

    accuracy = correct / len(test_cases)
    return accuracy
```

**Report:** "Name resolution accuracy: 95% (19/20 cases)"

---

### 4.2 Reaction Prediction Metrics

```python
def evaluate_reaction_prediction(test_cases):
    """
    test_cases = [
        {
            "reactants": "COc1ccccc1 . BrBr",
            "reagents": "Br[Fe](Br)Br",
            "expected_major": ["COc1ccc(Br)cc1"],
        },
        ...
    ]
    """
    top1_correct = 0
    top5_correct = 0
    mrr_sum = 0

    for case in test_cases:
        predictions = propose_products(case["reactants"], case["reagents"])
        pred_smiles = [canonicalize(p[0]) for p in predictions]
        expected = [canonicalize(s) for s in case["expected_major"]]

        # Top-1 accuracy
        if pred_smiles and pred_smiles[0] in expected:
            top1_correct += 1

        # Top-5 accuracy
        if any(s in expected for s in pred_smiles[:5]):
            top5_correct += 1

        # Mean Reciprocal Rank
        for i, pred in enumerate(pred_smiles):
            if pred in expected:
                mrr_sum += 1 / (i + 1)
                break

    n = len(test_cases)
    return {
        "top1_accuracy": top1_correct / n,
        "top5_accuracy": top5_correct / n,
        "mrr": mrr_sum / n,
    }
```

**Report:**
- "Top-1 accuracy: 75% (6/8 reactions)"
- "Top-5 accuracy: 87.5% (7/8 reactions)"
- "Mean Reciprocal Rank: 0.82"

---

### 4.3 Reference Lookup Metrics

```python
def evaluate_reference_lookup(test_smiles):
    """
    test_smiles = ["COc1ccccc1", "Cc1ccccc1", "c1ccccc1", ...]
    """
    found = 0
    total_peaks = 0

    for smiles in test_smiles:
        ref = get_reference_by_smiles(smiles)
        if ref:
            found += 1
            total_peaks += len(ref.get("ppm", []))

    coverage = found / len(test_smiles)
    avg_peaks = total_peaks / found if found > 0 else 0

    return {
        "coverage": coverage,
        "avg_peaks_per_compound": avg_peaks,
    }
```

**Report:**
- "Database coverage: 90% (9/10 compounds found)"
- "Average peaks per spectrum: 4.2"

---

### 4.4 Deconvolution Metrics

```python
import numpy as np

def evaluate_deconvolution(test_cases):
    """
    test_cases = [
        {
            "mixture_file": "binary_equal.csv",
            "ground_truth": {"toluene": 0.5, "benzene": 0.5},
            "references": [...],
        },
        ...
    ]
    """
    all_mae = []
    all_rmse = []

    for case in test_cases:
        # Load mixture
        ppm, intensity = load_csv(case["mixture_file"])

        # Run deconvolution
        result = quantify_single(ppm, intensity, case["references"])
        predicted = result["concentrations"]

        # Calculate errors
        errors = []
        for compound, true_conc in case["ground_truth"].items():
            pred_conc = predicted.get(compound, 0.0)
            errors.append(abs(true_conc - pred_conc))

        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(np.array(errors)**2))

        all_mae.append(mae)
        all_rmse.append(rmse)

    return {
        "mean_mae": np.mean(all_mae),
        "std_mae": np.std(all_mae),
        "mean_rmse": np.mean(all_rmse),
    }
```

**Report:**
- "Mean Absolute Error: 0.032 ± 0.015"
- "Root Mean Square Error: 0.041"

### 5.2 Results Section Template

```markdown
## Results

### Name Resolution Performance
The name resolution module correctly identified XX/YY (ZZ%) of
chemical names. Failures occurred for [list specific cases].

### Reaction Prediction Accuracy
Table 1 summarizes reaction prediction performance.

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 75.0% (6/8) |
| Top-5 Accuracy | 87.5% (7/8) |
| Mean Reciprocal Rank | 0.82 |

The model correctly predicted the major product for all
electrophilic aromatic substitution reactions. The single Top-1
failure was [specific case], where [explanation].

### Reference Database Coverage
Of the XX compounds queried, YY (ZZ%) were found in NMRBank.
Missing compounds included [list]. Found spectra contained an
average of N.N peaks per compound.

### Deconvolution Accuracy
Figure 1 shows predicted vs. true concentrations for all test
mixtures.

| Metric | Binary Mixtures | Ternary Mixtures | Overall |
|--------|-----------------|------------------|---------|
| MAE | 0.025 ± 0.012 | 0.041 ± 0.018 | 0.032 ± 0.015 |
| RMSE | 0.031 | 0.052 | 0.041 |
| R² | 0.97 | 0.94 | 0.96 |

Trace components (5% concentration) were successfully detected
in X/Y cases.

### Noise Robustness
Table 2 shows deconvolution accuracy as a function of noise level.

| Noise (σ) | MAE | Detection Rate |
|-----------|-----|----------------|
| 0.001 | 0.025 | 100% |
| 0.005 | 0.038 | 100% |
| 0.010 | 0.062 | 90% |
| 0.020 | 0.105 | 70% |
```

### 5.3 Discussion Points

```markdown
## Discussion

### Strengths
1. End-to-end automation from chemical names to quantification
2. No manual peak picking required
3. Handles multi-component mixtures

### Limitations
1. Reaction prediction limited to training data distribution
2. NMRBank coverage gaps for unusual compounds
3. Assumes linear mixing (may fail at high concentrations)

### Future Work
1. Expand reaction prediction to more reaction types
2. Incorporate ¹³C NMR for additional confirmation
3. Adaptive noise filtering for low-quality spectra
```

---

## 6. Running the Evaluation

### Quick Start (5 minutes)

```bash
# Generate test data
python -m app.eval.generate_test_data

# Run quick benchmark
python -m app.eval.run_all_benchmarks --quick --output-dir results/
```

### Full Evaluation

```bash
# 1. Generate all test data
python -m app.eval.generate_test_data --output-dir app/eval/data/synthetic

# 2. Run deconvolution benchmark
python -m app.eval.benchmark_deconvolution --output results/deconv.json

# 3. Run reaction prediction benchmark
python -m app.eval.benchmark_reactions --output results/reactions.json

# 4. Run NMRBank tests
pytest app/eval/test_nmrbank.py -v --tb=short

# 5. Run pipeline benchmark
python -m app.eval.benchmark_pipeline --output results/pipeline.json
```

### Interpreting Results

The JSON output files contain:

```json
{
  "summary": {
    "mae_mean": 0.032,
    "mae_std": 0.015,
    "top1_accuracy": 0.75,
    "coverage": 0.90
  },
  "results": [
    {
      "scenario_name": "binary_equal",
      "ground_truth": {"toluene": 0.5, "benzene": 0.5},
      "predicted": {"toluene": 0.48, "benzene": 0.52},
      "mae": 0.02
    },
    ...
  ]
}
```

---

## Appendix: Example Ground Truth Files

### Binary Mixture Ground Truth
```json
{
  "name": "binary_equal",
  "components": [
    {"name": "toluene", "smiles": "Cc1ccccc1", "fraction": 0.5},
    {"name": "benzene", "smiles": "c1ccccc1", "fraction": 0.5}
  ],
  "generation_params": {
    "noise_sigma": 0.001,
    "linewidth": 0.02,
    "resolution": 0.01
  }
}
```

### Reaction Test Case
```json
{
  "id": "eas_bromination_anisole",
  "name": "Bromination of anisole",
  "reactants": "COc1ccccc1 . BrBr",
  "reagents": "Br[Fe](Br)Br",
  "expected_major": ["COc1ccc(Br)cc1"],
  "expected_minor": ["COc1ccccc1Br"],
  "notes": "Para-substitution favored"
}
```
