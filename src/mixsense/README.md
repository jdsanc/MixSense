# 🤖 LLM-Enhanced Chemistry Analysis for NMR Hackathon

This folder contains an **LLM agentic workflow** that processes natural language requests and orchestrates specialized chemistry tools:

## 🧠 LLM Agent Workflow

1. **Parse Natural Language**: LLM understands your chemistry request
2. **Identify Species**: Converts chemical names to SMILES notation  i
3. **Predict Products**: Uses ReactionT5 for product prediction
4. **Find References**: Searches NMRBank for reference spectra
5. **Quantify Mixture**: Uses ASICS or Magnetstein for quantification
6. **Generate Narrative**: LLM creates a summary of results

## 🚀 Quick Start

```bash
# Set your HuggingFace token for LLM access
export HF_TOKEN=your_huggingface_token

# Launch the LLM-enhanced app
python launch_llm_app.py
```

## 💬 Example Natural Language Requests

- *"I want to analyze the bromination of anisole with Br2 using FeBr3 catalyst"*
- *"Help me quantify benzene and toluene in my mixture using robust analysis"* 
- *"Track the formation of p-bromoanisole over time from anisole bromination"*

> **Perfect for LLM Hackathons!** The LLM agent intelligently routes requests to specialized chemistry tools while providing natural language interaction.

## 🏗️ Architecture Overview

```
User Input: "Analyze bromination of anisole with Br2 and FeBr3"
    ↓
LLM Agent (llm_agent.py)
    ↓
PocketFlow Workflow (agent_pocketflow.py)
    ↓
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   ParseNL       │ NormalizePropose│ RetrieveRefs    │ Quantify        │
│   (LLM)         │ (ReactionT5)    │ (NMRBank)       │ (ASICS/Magnet)  │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘
    ↓
Narrate (LLM) → Results + Narrative
```

**Key Components:**
- **`llm_agent.py`**: Core LLM agent for natural language processing
- **`agent_pocketflow.py`**: Workflow orchestration using PocketFlow
- **`gradio_llm_app.py`**: Enhanced UI with natural language input
- **`tools_*.py`**: Existing chemistry tools (unchanged)

---

## Features

### 🤖 LLM-Enhanced Features (NEW!)
* **Natural Language Input**: Describe your chemistry analysis in plain English
* **Intelligent Tool Orchestration**: LLM automatically selects and sequences tools
* **Automatic Backend Selection**: Chooses ASICS vs Magnetstein based on request
* **Narrative Generation**: LLM creates summaries of analysis results
* **Workflow Tracking**: See each step of the agent's decision process

### 🔧 Core Analysis Features
* **Single-sample** tab: reactants → products → select species → quantify mixture → concentrations + overlay.
* **Backend switch**: drop-down to choose **ASICS** or **Magnetstein**.
* **Time-series** tab (Magnetstein): upload multiple CSVs (e.g., `t0.csv`, `t5.csv`, …) → proportions vs time.
* **(Optional) NMR→Structure** block: run `onmt_translate` against a provided checkpoint + tokenized `src.txt`.
* **Pluggable tools** (`app/tools_*.py`) so you can swap/extend components easily.

---

## File map

```
app/
  README.md                    ← this file
  gradio_llm_app.py            ← 🆕 LLM-Enhanced UI (Natural language input + agent workflow)
  gradio_app.py                ← Original UI (Manual tool orchestration)
  llm_agent.py                 ← 🆕 Core LLM agent for natural language processing
  agent_pocketflow.py          ← 🆕 PocketFlow workflow orchestration (LLM + tools)
  agent.py                     ← Original orchestrator (propose → refs → quantify)
  tools_reactiont5.py          ← wrapper around ReactionT5v2-forward (HF)
  tools_nmrbank.py             ← loads NMRBank refs (ppm,intensity) by SMILES
  tools_asics.py               ← bridge to R/ASICS
  tools_magnetstein.py         ← (optional) Magnetstein wrappers (single + time-series)
  tools_nmr2structure.py       ← (optional) OpenNMT inference wrapper for NMR→Structure
  test_llm_integration.py      ← 🆕 Test script for LLM agent functionality
  examples/
    generate_demo_csv.py       ← makes synthetic single/time-series CSVs from NMRBank refs
launch_llm_app.py              ← 🆕 Easy launcher for LLM-enhanced app
r/
  asics_quantify.R             ← called by tools_asics.py
external/
  NMRExtractor/                ← (optional) submodule; contains NMRBank/
```

---

## Requirements

* **Python** ≥ 3.10
* **R** (for ASICS path; 4.x recommended) + ability to run `Rscript`
* **Conda** recommended for RDKit

**Python packages (core):**

* `rdkit` (conda-forge), `transformers`, `accelerate`, `torch`, `gradio`, `pandas`, `numpy`, `huggingface_hub`, `requests`

**LLM Agent Requirements:**

* **HuggingFace Token**: `HF_TOKEN` environment variable for LLM API access
* **pocketflow**: For workflow orchestration (required for LLM agent)

**Optional:**

* **Magnetstein** (quant backend): install package or add as a git submodule; optional `gurobipy` can speed up solves.
* **OpenNMT-py** (`onmt_translate`) for NMR→Structure.

### Suggested env setup

```bash
# Create env
conda create -y -n llmchem python=3.10
conda activate llmchem

# Core deps
conda install -y -c conda-forge rdkit r-base r-essentials
pip install "transformers>=4.44" "accelerate>=0.30" torch gradio pandas numpy huggingface_hub requests

# LLM Agent requirements
pip install pocketflow

# Optional: OpenNMT + (if you plan to try it) Magnetstein solver
pip install onmt-py

# (Optional) If Magnetstein requires it and you have a license:
# pip install gurobipy

# Set HuggingFace token for LLM access
export HF_TOKEN=your_huggingface_token_here
```

### ASICS (R)

Ensure the **ASICS** R package is available to `Rscript`:

```bash
R -q -e 'install.packages("remotes"); remotes::install_github("sipss/ASICS")'
# or if on CRAN in your setup:
# R -q -e 'install.packages("ASICS")'
```

---

## NMRBank data (choose one)

1. **HF dataset**: `sweetssweets/NMRBank`

```python
from huggingface_hub import snapshot_download
snapshot_download("sweetssweets/NMRBank", repo_type="dataset",
                  local_dir="NMRBank", local_dir_use_symlinks=False)
```

CLI:

```bash
huggingface-cli download sweetssweets/NMRBank --repo-type dataset \
  --local-dir ./NMRBank --local-dir-use-symlinks False
```

2. **Git submodule** (repo contains `NMRBank/`):

```bash
git submodule add -b main https://github.com/eat-sugar/NMRExtractor.git external/NMRExtractor
git submodule update --init --recursive
# optional: only check out the NMRBank folder
(cd external/NMRExtractor && git sparse-checkout init --cone && git sparse-checkout set NMRBank)
```

**Tell the app where the data is** (if not using the default path):

```bash
export NMRBANK_DIR=/absolute/path/to/NMRBank
```

`tools_nmrbank.py` checks `NMRBANK_DIR`, then falls back to `../external/NMRExtractor/NMRBank` or `../NMRBank` relative to `app/`.

---

## Running the app

### 🤖 LLM-Enhanced App (Recommended)

```bash
# Easy launcher (checks environment and starts app)
python launch_llm_app.py

# Or run directly
conda activate llmchem
export HF_TOKEN=your_huggingface_token
python -m app.gradio_llm_app
```

**LLM Agent Workflow:**

1. **Describe your analysis** in natural language (e.g., "Analyze bromination of anisole with Br2")
2. **Upload mixture CSV** (optional) with two columns: `ppm,intensity`
3. **Click "Run LLM Analysis"** → Agent automatically:
   - Parses your request
   - Identifies reactants and reagents
   - Proposes products using ReactionT5
   - Finds reference spectra from NMRBank
   - Quantifies mixture using ASICS or Magnetstein
   - Generates narrative summary
4. **View results** in the analysis table and narrative

### 🔧 Original Manual App

```bash
# From repo root (env activated)
python -m app.gradio_app
# Gradio prints a local URL; open it in your browser.
```

**Single sample (typical flow):**

1. Enter **Reactants** (SMILES or simple names, comma-separated) and optional **Reagents**.
2. Click **Propose products** → products list appears; table shows which species have library refs.
3. Choose **Quantification backend** (ASICS or Magnetstein).
4. Upload a **mixture CSV** with two columns: `ppm,intensity`.
5. (Optional) Enter **row indices** to restrict which species to fit.
6. Click **Quantify** → concentrations table + overlay appear.

**Time-series (Magnetstein):**

* Provide a **species list** (SMILES or names).
* Upload multiple **CSV files** (two columns each). The app parses **time** from filenames (first number found).
* Click **Quantify time-series** → proportions-vs-time table/JSON appears.

**NMR→Structure (optional):**

* Check "Enable NMR→Structure", then select an **OpenNMT checkpoint (`.pt`)** and a **tokenized `src.txt`** (one spectrum per line in their token format).
* Press **Run** → shows top predictions (first input only, for demo).

---

## Demo data (quick start)

Generate synthetic spectra from NMRBank references:

```bash
python -m app.examples.generate_demo_csv
# → app/examples/anisole_bromination_demo.csv            (single sample)
# → app/examples/bromination_series/t0.csv ... t30.csv   (time series)
```

Then in the UI:

* **Single sample:** upload `anisole_bromination_demo.csv`.
* **Time-series:** multi-select and upload all CSVs in `bromination_series/`.

---

## Troubleshooting

### LLM Agent Issues
* **"HF_TOKEN not set"** → Set `export HF_TOKEN=your_huggingface_token`
* **"LLM Error"** → Check your HuggingFace token and internet connection
* **"PocketFlow import error"** → Install with `pip install pocketflow`
* **Agent workflow fails** → Check that all tool dependencies are installed

### General Issues
* **"Magnetstein not available"** → install the package or add its source; or select **ASICS**.
* **ASICS errors** → ensure `Rscript` is on `PATH` and ASICS is installed; check `r/asics_quantify.R`.
* **No references found** → confirm `NMRBANK_DIR` and that species SMILES exist in the library (canonicalization matters).
* **HF downloads slow** → `export HF_HUB_ENABLE_HF_TRANSFER=1`.
* **OpenNMT not found** → install `onmt-py` so `onmt_translate` is on PATH.

### Testing
```bash
# Test the LLM integration
python app/test_llm_integration.py

# Test with conda environment
conda activate llmchem
python app/test_llm_integration.py
```

---

## Notes & Credits

* **Reaction proposals:** `sagawa/ReactionT5v2-forward` (Hugging Face).
* **ASICS** (R), **Magnetstein** (OT-based quantification), **rxn4chemistry/nmr-to-structure** (OpenNMT).
* **NMRBank** via `sweetssweets/NMRBank` or the `NMRBank/` folder in `eat-sugar/NMRExtractor`.

*Intended for hackathon/showcase use; review dependency licenses before commercial use.*
