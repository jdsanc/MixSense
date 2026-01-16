# 🧪 NMR Chemistry Analysis Platform

NMR (Nuclear Magnetic Resonance) chemistry analysis platform. Provides tools for reaction product prediction, NMR reference lookup, and mixture quantification.

## Quick Start

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

### Run Tests
```bash
pytest tests/ -v

# Skip NMRBank CSV loading (faster)
NMRBANK_SKIP_LOAD_FOR_TESTS=1 pytest tests/ -v
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
| `GRB_LICENSE_FILE` | Gurobi license file path |
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
