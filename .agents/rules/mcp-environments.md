---
trigger: always_on
---

# Conda Environment Rules

When running scripts, always activate the Conda environment documented in the script's `# Env:` annotation or Requirements docstring.

For general utilities (image processing, plotting, metadata extraction) use `base-agent`. For NMR simulation and mixture deconvolution use `nmr-agent`.

## Installation Instructions

For detailed installation instructions, refer to the `README.md` and `install.sh` in each environment's directory under `conda-envs/<env_name>/`.

- **base-agent**: `conda-envs/base-agent/` (Core: opencv, scikit-image, numpy, scipy, matplotlib, python-dotenv, Pillow, google-genai, openai)
- **nmr-agent**: `conda-envs/nmr-agent/` (Core: nmrsim, rdkit, numpy, scipy, matplotlib, requests, scikit-learn)

## Environment to Script Mapping

| Environment | Skills / Scripts |
| :--- | :--- |
| `base-agent` | `chem-plot-digitizer` (all scripts), general utilities |
| `nmr-agent` | `chem-nmr-predict`, `chem-nmr-analysis` |
