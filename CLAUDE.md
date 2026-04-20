# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MixSense is an AI-agent-assisted framework for quantifying chemical mixtures from crude 1H NMR spectra. Given a reaction description and spectrum (image or numeric), it digitizes spectra, predicts reaction products from SMILES, generates simulated reference spectra, and performs Wasserstein-distance LP deconvolution to extract mole fractions.

## Environment Setup

Single conda environment for all scripts:

```bash
bash conda-envs/mixsense/install.sh
```

Run scripts with:
```bash
conda run -n mixsense python <script> [args]
```

## Agent Framework

### Skill Discovery
```bash
grep -r "^description:" .agents/skills/*/SKILL.md
```

Check `.agents/workflows/` first for end-to-end protocols, then find the relevant skill(s) in `.agents/skills/`.

### Workflow for Research Tasks
1. Create research dir: `research/<date>_<short_description>/` (use `utils/research_utils.py`)
2. Check `.agents/workflows/` for an applicable end-to-end protocol
3. Map to individual skills; run scripts with correct `# Env:` environment
4. Visually inspect all generated plots before proceeding

## Architecture

```
.agents/
  workflows/    ← end-to-end research protocols (chain multiple skills)
  skills/       ← composable capabilities (each has SKILL.md + scripts/ + examples/)
  rules/        ← governance: skill-standards, coding-standards, plot-standards, workflow-standards
conda-envs/     ← reproducible environment definitions
utils/          ← research_utils.py (dir management), paper_downloader.py
research/       ← all run outputs (gitignored)
```

### Key Skills
- `digitize_plot` MCP tool — image → (ppm, intensity) numeric data (replaces chem-plot-digitizer skill)
- `nmr-predict` — SMILES → predicted 1H NMR via SPINUS API + nmrsim simulation
- `nmr-analysis` — Wasserstein LP deconvolution, kinetics, spectrum plotting
- `drug-db-pubchem` — compound name/structure lookup via PubChem PUG-REST API

### Deconvolution Quality
Wasserstein distance (WD) thresholds: `< 0.05` good, `0.05–0.15` acceptable, `> 0.15` poor. Key param: `--protons` (number of 1H per molecule for mole fraction normalization), `--kappa` regularization (default 0.25).

## Coding Standards

- Every bash block in skills/workflows must include `# Env: <env-name>`
- No `try/except` unless output is inherently non-deterministic; code should be predictable
- Absolute imports only (e.g., `from spectra import load_spectrum`)
- Spectrum files: `.csv` (comma), `.xy` (tab), `.tsv` (tab) — auto-detected by extension
- All outputs: `.svg` + `.png`; figures 6×5 (landscape) or 5×5 (square); `linewidth=2.5`; bold axis labels; `tight_layout()`
- New skills: kebab-case with category prefix (`chem-`, `drug-`, `ml-`, `mat-`, `general-`); follow `.agents/rules/skill-standards.md`

## Rules Reference
Full standards in `.agents/rules/`: `skill-standards.md`, `coding-standards.md`, `plot-standards.md`, `workflow-standards.md`.
