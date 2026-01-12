# MixSense (LLMHackathon) – LLM‑orchestrated NMR mixture analysis and property prediction

This repo implements the MixSense workflow: an LLM‑driven agent that ingests ¹H‑NMR spectra and natural‑language queries, proposes plausible reaction products, retrieves or simulates pure‑component references, deconvolves mixtures to quantify components (single sample and time series), and optionally runs NMR→structure inference.


## Overview

- User query → LLM parses intent and species
- Product hypothesis → ReactionT5 (local or HF API)
- Reference retrieval → NMRBank CSV or NMRDB.org
- Quantification → ASICS (R) or Magnetstein (optimal transport)
- Time series → Magnetstein across multiple spectra
- Narrative → LLM summarizes outcomes
- Optional → NMR→Structure via OpenNMT


## Folder map (only relevant pieces)

- `launch_llm_app.py` – launcher that checks env and starts the Gradio app.
- `app/` – all MixSense agent logic and tools:
	- `gradio_llm_app.py` – LLM‑enhanced UI. Natural language in, agent pipeline out. Exposes tabs for single‑sample and time‑series.
	- `tool_gradio_app.py` – classic tool UI (manual: propose → refs → quantify, plus optional NMR→structure). Useful if you want less LLM in the loop.
	- `llm_agent.py` – ChemistryLLMAgent. Parses requests, proposes products, loads references, runs quantification, and generates narrative. Key entries:
		- `ChemistryLLMAgent.parse_chemistry_request`
		- `ChemistryLLMAgent.execute_analysis`
		- `process_natural_language_request` and `get_agent`
	- `agent_pocketflow.py` – builds a PocketFlow graph: parse → normalize+propose → retrieve refs → quantify → narrate. Used by `gradio_llm_app.py`.
	- `agent.py` – shared utilities:
		- `normalize_smiles_list` (names→SMILES, with local helpers)
		- `step_propose` (ReactionT5 wrapper)
		- `load_refs_for_species` (NMRBank→NMRDB.org fallback)
		- `quantify_mixture` (ASICS path)
	- Quantification backends:
		- `tools_asics.py` – ASICS wrapper (R). Resamples refs to mixture grid, calls `r/asics_quantify.R`, returns `{"components": [{name,fraction}], "raw": ...}`.
		- `tools_magnetstein.py` – Magnetstein wrappers:
			- `quantify_single(ppm,intensity,library)` → `{"concentrations": {name: value}, "reconstructed": {...}}`
			- `quantify_timeseries(times, mixtures, library)` → time‑indexed proportions.
		- `tool_deconvolve_nmr.py` – CLI deconvolution using Masserstein+Gurobi (prints a table and optional JSON). Called only if you explicitly choose the `deconvolve` path.
	- Reference sources:
		- `tools_nmrbank.py` – loads NMRBank data from CSV; returns `{name,smiles,ppm,intensity}`.
		- `tools_nmrdb.py` – NMRDB.org fetch/parse helpers: `get_nmr_spectrum_for_smiles`, `get_reference_by_smiles_nmrdb`, `create_magnetstein_library_from_smiles`, plus simple tests.
	- Reaction prediction:
		- `tools_reactiont5.py` – prefers the local model in `reactants_to_products/`, else falls back to Hugging Face Inference API (`sagawa/ReactionT5v2-forward`). Also provides `get_unique_components` to dedup dot‑separated products into species.
	- Optional structure inference:
		- `tools_nmr2structure.py` – thin wrapper over `onmt_translate` for rxn4chemistry/nmr‑to‑structure.
	- Examples/tests:
		- `examples/generate_demo_csv.py` – demo data generator
		- `test_llm_integration.py` – quick sanity tests for parsing, graph building and CSV parsing.
- `r/asics_quantify.R` – Rscript entry point for ASICS quantification.
- `magnetstein/` – vendor subpackage containing `masserstein` and helpers. Installable (`pip install -e magnetstein`).
- `NMRBank/` – zipped CSV/JSON datasets. See preparation steps below.
- `reactants_to_products/` – local ReactionT5 helpers and post‑processing.
- `nmr-to-structure/` – standalone subpackage for spectrum simulation and NMR→structure (see its README for usage).
- `envs/` – environment YAMLs you can use as a starting point.
- `scripts/test_asics.py` – toy ASICS quantification demo.
- `run_nmr_analysis.sh` – legacy runner that calls `LLM_code_for_tool/llama_tool.py` (currently a placeholder); not required for the MixSense app.


## Setup

### 1) Python packages

Install core runtime. Minimal pip set:

```bash
pip install gradio pandas numpy requests rdkit-pypi pocketflow beautifulsoup4
# optional backends
pip install pulp  # for Magnetstein default solver
```

Then install the vendor Magnetstein package in editable mode:

```bash
pip install -e magnetstein
```

If you want the local ReactionT5 path (faster, offline):

```bash
pip install transformers torch
```

For optional NMR→structure:

```bash
pip install OpenNMT-py
```

### 2) R + ASICS (for the ASICS backend)

- Install R and the ASICS package.
- Make sure `Rscript` is on your PATH.
- The Python wrapper calls `r/asics_quantify.R` directly.

### 3) NMRBank data

`tools_nmrbank.py` loads a CSV at:

```
NMRBank/NMRBank/NMRBank_data_with_SMILES_156621_in_225809.csv
```

Preparation:

1) Unzip `NMRBank_data_with_SMILES_156621_in_225809.zip` into a new `NMRBank/NMRBank/` folder so the CSV sits at the path above.
2) If you prefer the JSON set, adjust `tools_nmrbank.py` accordingly (the repo currently expects CSV).

If a species isn’t found in NMRBank, the agent can fall back to NMRDB.org via `tools_nmrdb.py`.

### 4) Credentials

Set your Hugging Face token for the LLM and ReactionT5 API fallback:

```bash
export HF_TOKEN=your_huggingface_token
```


## How to run

### Option A: LLM‑enhanced app (recommended)

```bash
python LLMHackathon/launch_llm_app.py
```

This starts the Gradio app defined in `app/gradio_llm_app.py` with:

- Tab “LLM Agent Analysis”: enter a natural‑language request and optionally upload a mixture CSV (`ppm,intensity`). The agent builds the workflow with PocketFlow and returns a narrative + results table.
- Tab “Time‑Series Analysis”: upload multiple CSVs (filenames should contain a time value), and the Magnetstein backend tracks proportions vs time.

Example queries:

- “Analyze bromination of anisole with Br₂ using FeBr₃.”
- “Quantify benzene and toluene in my mixture.”
- “Track p‑bromoanisole formation over time.”

### Option B: Classic tool UI

```bash
python -m app.tool_gradio_app
```

Manual steps in the UI:

1) Propose products from reactants/reagents → unique components.
2) Load references (from NMRBank and/or NMRDB).
3) Quantify with backend choice:
	 - ASICS (R)
	 - Magnetstein (optimal transport)
4) Optional: NMR→Structure with OpenNMT.

### Command‑line quick tests

```bash
python app/test_llm_integration.py
python scripts/test_asics.py
```


## Data contracts and shapes

- Mixture CSV: two columns `[ppm,intensity]` (no header names required; the app renames columns internally).
- Reference spectra: list of dicts `{name, ppm: [...], intensity: [...]}` on a common ppm grid. ASICS resamples automatically; Magnetstein requires non‑negative ppm values.
- Quantification outputs:
	- ASICS: `{components: [{name, fraction}], raw: {...}}`
	- Magnetstein: `{concentrations: {name: value}, reconstructed: {ppm, intensity}}`


## Relevant scripts and their role

- `app/gradio_llm_app.py` – High‑level, agent‑driven UI. Calls `agent_pocketflow.build_llm_graph` and `llm_agent.get_agent()`.
- `app/tool_gradio_app.py` – Low‑level tool UI with explicit controls; supports time‑series and NMR→structure.
- `app/llm_agent.py` – Core agent implementation (DeepSeek default via Together router). Contains robust JSON parsing fallback and narrative generation.
- `app/agent_pocketflow.py` – Orchestrates nodes with PocketFlow; includes safe function node wrapper and backends.
- `app/agent.py` – SMILES normalization, product proposal, reference loading, and ASICS quantification helper.
- `app/tools_asics.py` + `r/asics_quantify.R` – R backend glue. Ensure R/ASICS installed.
- `app/tools_magnetstein.py` – Optimal transport backends for single sample and time series; uses `masserstein` (installed via `magnetstein/`).
- `app/tools_nmrbank.py` – CSV loader for NMRBank (requires unzip step).
- `app/tools_nmrdb.py` – Web fetch and parsing from NMRDB.org with multiple fallback strategies; includes small tests.
- `app/tools_reactiont5.py` – Product SMILES via local model or HF API; provides `get_unique_components`.
- `app/tools_nmr2structure.py` – OpenNMT wrapper for structure prediction.
- `app/test_llm_integration.py` – Smoke tests for parsing & graph.
- `scripts/test_asics.py` – Toy synthetic spectrum demo for ASICS.
- `tool_deconvolve_nmr.py` – Standalone CLI deconvolution using Gurobi; optional path for power users.


## Notes and troubleshooting

- LLM calls fail → ensure `HF_TOKEN` is exported.
- ASICS errors → verify R and ASICS installation; confirm `Rscript` is on PATH.
- NMRBank missing CSV → unzip into `NMRBank/NMRBank/` so `tools_nmrbank.py` can find the file.
- Magnetstein import errors → run `pip install -e magnetstein` and install `pulp` (CBC solver) or configure Gurobi if using the CLI.
- NMRDB rate limits → `tools_nmrdb.py` sleeps between requests; adjust `delay` if needed.


## Attribution

Implements the MixSense workflow outlined in the attached figures: mixture identification, quantification (single/time‑series), and optional property/structure prediction, with an all‑in‑one UI and LLM‑orchestrated pipeline.
