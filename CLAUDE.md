# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NMR (Nuclear Magnetic Resonance) chemistry analysis platform. Provides tools for reaction product prediction, NMR reference lookup, and mixture quantification.

## Setup

```bash
uv sync
```

## Common Commands

### Run the Gradio UI
```bash
uv run python -m app.gradio_llm_app
# Launches at localhost:7667
```

### Run the MCP Server
```bash
uv run python -m app.mcp_server
# Exposes chemistry tools via Model Context Protocol
```

### Run NMR Deconvolution CLI
```bash
uv run python app/tool_deconvolve_nmr.py \
  mixture.csv comp1.csv comp2.csv \
  --protons 16 12 \
  --names "Component 1" "Component 2" \
  --json
```

### Run Tests
```bash
uv run pytest tests/ -v

# Skip NMRBank CSV loading (faster)
NMRBANK_SKIP_LOAD_FOR_TESTS=1 uv run pytest tests/ -v
```


## Architecture

### MCP Server (app/mcp_server.py)
Exposes chemistry tools via Model Context Protocol:
- `lookup_nmr_reference(smiles)` - Find NMR reference by SMILES
- `resolve_chemical_name(name)` - Convert name to SMILES
- `predict_products(reactants, reagents)` - Predict reaction products
- `deconvolve_mixture(ppm, intensity, refs)` - Quantify mixture components
- `load_references_for_smiles(smiles_list)` - Batch reference lookup

### Core Tools (app/tools_*.py)

**tools_nmrbank.py** - NMRBank reference lookup. Lazy-loads CSV into LUT keyed by canonical SMILES.

**tools_reactiont5.py** - Product prediction using ReactionT5 model. Local model preferred, falls back to HuggingFace API.

**tools_deconvolve.py** - Wrapper around the deconvolution CLI script.

**tools_magnetstein.py** - Quantification using magnetstein's optimal transport (CBC solver, no license required).

### Simple Agent (app/simple_agent.py)
Straightforward implementation that chains tools:
1. Resolve reactant names → SMILES
2. Predict products
3. Find reference spectra
4. Optionally quantify mixture

### Gradio UI (app/gradio_llm_app.py)
Web interface using tools directly.

### Submodules
- **magnetstein/** - Masserstein optimal transport library
- **NMRBank/** - Reference NMR spectra database (~156k compounds)

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace API token (required for reaction prediction via API) |
| `NMRBANK_CSV` | Override NMRBank CSV location |
| `NMRBANK_SKIP_LOAD_FOR_TESTS` | Set to `1` to skip CSV load in tests |

## Data Formats

**NMR Spectra CSV**: Two columns (ppm, intensity), no header. TSV also supported.

**Reference Library**: List of dicts: `{name, ppm: [...], intensity: [...], smiles}`

## Dependencies

Managed via `pyproject.toml` + `uv.lock`. Run `uv sync` to install.

Key packages:
- rdkit, transformers, torch (for ReactionT5)
- IsoSpecPy, masserstein (for deconvolution)
- mcp (for MCP server)
- gradio (for web UI)
- pulp (CBC solver for deconvolution, no license required)
