---
trigger: manual
---

# MixSense: NMR Mixture Analysis Agent Guide

## Project Overview

MixSense is an AI-agent-assisted framework for quantifying chemical mixtures from crude 1H NMR spectra. Given a reaction description and a 1H NMR spectrum (numeric data or an image), the agent identifies all species present, generates simulated reference spectra, and performs Wasserstein-distance deconvolution to extract mole fractions.

The framework consists of the following components:

## Core Capabilities

- **Spectrum digitization**: Extract numeric (ppm, intensity) data from NMR spectrum images using the `chem-plot-digitizer` skill.
- **Product prediction**: Predict reaction products from SMILES strings via the `chem-nmr-analysis` skill (`get_products.py`, powered by ReactionT5).
- **Reference spectrum generation**: Predict 1H NMR spectra for all species via the `chem-nmr-predict` skill.
- **Mixture deconvolution**: Quantify mole fractions via Wasserstein-distance LP deconvolution (`chem-nmr-analysis`, `deconvolve.py`).
- **Kinetics analysis**: Extract mole-fraction-vs-time curves from time-series NMR spectra (`chem-nmr-analysis`, `kinetics.py`).

## Environment Setup

Two conda environments are used:

| Environment | Purpose |
|---|---|
| `base-agent` | Image processing, plot digitization, general utilities |
| `nmr-agent` | NMR simulation (nmrsim), deconvolution, kinetics |

Install via:
```bash
bash conda-envs/base-agent/install.sh
bash conda-envs/nmr-agent/install.sh
```

## Step-by-Step Workflow

See `.agents/workflows/reaction-to-nmr-quantification.md` for the end-to-end protocol, and `.agents/workflows/nmr-reaction-kinetics.md` for time-series kinetics.

## Framework Structure

- **Skills** (`.agents/skills/`): Mid-level tutorials combining scripts to solve focused tasks. Each has a `SKILL.md` with step-by-step instructions.
- **Workflows** (`.agents/workflows/`): End-to-end research protocols that chain multiple skills.

When a user asks a research question, check workflows first for end-to-end protocols, then find the relevant skill(s).
