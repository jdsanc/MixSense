---
name: chem-nmr-mixture-analysis
description: Quantify components in 1H-NMR spectra of reaction mixtures via supervised Wasserstein deconvolution against reference spectra — either for a single time point or across a reaction time series to extract kinetics.
category: chemistry
---

# NMR Mixture Analysis

## Goal

Determine the molar composition of a 1H-NMR mixture spectrum by deconvolving it against reference spectra for each known component.

The full workflow starts from a **reaction description** and requires no prior knowledge of what products formed:

1. **Predict candidates**: resolve reactant names to SMILES, predict products/impurities with ReactionT5, look up their 1H NMR reference spectra from NMRBank — all automated.
2. **Deconvolve**: fit a crude mixture spectrum against those references to get mole fractions.

If you already have reference spectra on hand (measured or digitized), you can skip straight to Step 2.

Two deconvolution use cases, both supervised:

- **Single time point**: one crude spectrum + references -> estimated mole fractions + Wasserstein distance (fit quality).
- **Reaction kinetics (time series)**: crude spectra at multiple times + same references -> mole fraction vs time table and kinetics plot.

All input files are two-column `.csv` or `.xy` (ppm, intensity), as produced by spectrum digitizers or NMR software (e.g. Mnova).

---

## Instructions

### Step 1 — Predict candidate components and look up reference spectra

Write a plain-text reaction description (see `examples/reaction.txt`):

```
Reduction of camphor with NaBH4
```

Or structured format:
```
Reactants: camphor
Reagents: NaBH4, methanol
```

Then run the candidate prediction script:

```bash
# Env: mixsense
export HF_TOKEN=your_huggingface_token
export NMRBANK_CSV=/path/to/nmrbank.csv

uv run python mixsense-skills/chem-nmr-mixture-analysis/scripts/predict_candidates.py \
  --reaction reaction.txt \
  --output_dir candidates/
```

Or pass names directly:

```bash
uv run python mixsense-skills/chem-nmr-mixture-analysis/scripts/predict_candidates.py \
  --reactants "camphor" \
  --reagents "NaBH4" \
  --output_dir candidates/
```

What this does:
1. Resolves each name to canonical SMILES (RDKit direct parse, then PubChem REST API — no token needed for this step)
2. Calls the [ReactionT5](https://huggingface.co/sagawa/ReactionT5v2-forward) model via HuggingFace API to predict reaction products and impurities (`HF_TOKEN` required)
3. Looks up 1H NMR reference spectra for all species (reactants + predicted products) in NMRBank
4. Saves each found spectrum as a CSV file in `candidates/`
5. Prints a `candidates/candidates.json` manifest and a ready-to-run deconvolution command

Output of `candidates/`:
- `<compound_name>.csv` — one file per found reference spectrum (ppm, intensity)
- `candidates.json` — manifest: reactants, reagents, predicted products, found/missing references

> **If NMRBank has no entry** for a predicted species, the script lists it under `missing_references`. Provide a measured or digitized reference CSV for those manually.

> **To skip product prediction** (reactants only, or if you have no HF_TOKEN):
> ```bash
> uv run python ... --skip_prediction
> ```

---

### Step 2 — Inspect spectra visually

Before deconvolving, overlay all spectra to spot calibration offsets or baseline drift:

```bash
# Env: mixsense
uv run python mixsense-skills/chem-nmr-mixture-analysis/scripts/plot.py \
  crude.csv candidates/borneol.csv candidates/isoborneol.csv \
  --labels "Mixture" "borneol" "isoborneol" \
  --title "Mixture vs. References" \
  --output results/spectra_overview.png
```

---

### Step 3a — Single time point: supervised deconvolution

Use the reference CSVs from Step 1 (or your own):

```bash
# Env: mixsense
uv run python mixsense-skills/chem-nmr-mixture-analysis/scripts/deconvolve.py \
  crude.csv candidates/borneol.csv candidates/isoborneol.csv \
  --protons 18 18 \
  --names "borneol" "isoborneol" \
  --baseline-correct \
  --plot results/deconvolution_result.png \
  --json
```

Key arguments:
- `--protons 18 18`: number of 1H protons per molecule for each component. Required for accurate molar fractions. Must match the order of reference files.
- `--names`: human-readable labels for output and plot.
- `--baseline-correct`: shifts each spectrum so its minimum intensity = 0. Recommended for digitized or NMRBank spectra with a non-zero baseline.
- `--plot results/result.png`: saves a stacked figure showing mixture, each scaled component, the reconstructed fit, and the residual.
- `--json`: also prints a machine-readable JSON result line.

Interpreting output:
```
Estimated proportions:
  borneol      0.20
  isoborneol   0.80

Wasserstein distance: 0.001234
```
The **Wasserstein distance** is the fit quality metric — lower is better. Values below ~0.01 indicate a good fit for well-calibrated spectra. Values above ~0.05 suggest a calibration mismatch or missing component.

---

### Step 3b — Time series: reaction kinetics

When crude spectra are available at multiple time points, run deconvolution at each point and collect results into a kinetics table and plot:

```bash
# Env: mixsense
uv run python mixsense-skills/chem-nmr-mixture-analysis/scripts/kinetics.py \
  --refs candidates/borneol.csv candidates/isoborneol.csv \
  --timepoints t000min.csv t010min.csv t020min.csv t030min.csv t060min.csv \
  --times 0 10 20 30 60 \
  --time_unit min \
  --protons 18 18 \
  --names "borneol" "isoborneol" \
  --baseline_correct \
  --output_dir results/kinetics
```

Outputs (in `--output_dir`):
- `kinetics.csv`: table of time, mole fractions per component, Wasserstein distance at each time point.
- `kinetics_plot.png`: mole fraction vs time (top) + Wasserstein distance vs time (bottom, fit quality indicator).

---

## Examples

### Example A — Single time point: camphor NaBH4 reduction

#### A1 — With automated candidate prediction (full workflow)

```bash
# Env: mixsense
export HF_TOKEN=your_token
export NMRBANK_CSV=/path/to/nmrbank.csv

uv run python mixsense-skills/chem-nmr-mixture-analysis/scripts/predict_candidates.py \
  --reaction mixsense-skills/chem-nmr-mixture-analysis/examples/reaction.txt \
  --output_dir mixsense-skills/chem-nmr-mixture-analysis/examples/candidates/

# Then deconvolve with the predicted references:
uv run python mixsense-skills/chem-nmr-mixture-analysis/scripts/deconvolve.py \
  mixsense-skills/chem-nmr-mixture-analysis/examples/deconvolution/crude.csv \
  mixsense-skills/chem-nmr-mixture-analysis/examples/candidates/borneol.csv \
  mixsense-skills/chem-nmr-mixture-analysis/examples/candidates/isoborneol.csv \
  --names "borneol" "isoborneol" \
  --baseline-correct \
  --plot mixsense-skills/chem-nmr-mixture-analysis/examples/results/deconvolution_result.png \
  --json
```

#### A2 — With hand-digitized references (skip prediction)

Deconvolve directly against borneol and isoborneol reference spectra digitized from Dhawan et al., 2022:

```bash
# Env: mixsense
uv run python mixsense-skills/chem-nmr-mixture-analysis/scripts/deconvolve.py \
  mixsense-skills/chem-nmr-mixture-analysis/examples/deconvolution/crude.csv \
  mixsense-skills/chem-nmr-mixture-analysis/examples/deconvolution/borneol.csv \
  mixsense-skills/chem-nmr-mixture-analysis/examples/deconvolution/isoborneol.csv \
  --protons 18 18 \
  --names "borneol" "isoborneol" \
  --baseline-correct \
  --plot mixsense-skills/chem-nmr-mixture-analysis/examples/results/deconvolution_result.png \
  --json
```

Expected: isoborneol is the major product (~80%), consistent with the known stereoselectivity of NaBH4 reduction of camphor (literature: ~1:4 borneol:isoborneol).

### Example B — Reaction kinetics: simulated camphor reduction time series

Synthetic crude spectra at 8 time points generated by linearly combining the borneol and isoborneol references with a 1st-order kinetic profile (borneol -> isoborneol, k = 0.04 min^-1, asymptote 20:80). Ground truth stored in `examples/kinetics/`.

```bash
# Env: mixsense
uv run python mixsense-skills/chem-nmr-mixture-analysis/scripts/kinetics.py \
  --refs \
    mixsense-skills/chem-nmr-mixture-analysis/examples/deconvolution/borneol.csv \
    mixsense-skills/chem-nmr-mixture-analysis/examples/deconvolution/isoborneol.csv \
  --timepoints \
    mixsense-skills/chem-nmr-mixture-analysis/examples/kinetics/t000min.csv \
    mixsense-skills/chem-nmr-mixture-analysis/examples/kinetics/t005min.csv \
    mixsense-skills/chem-nmr-mixture-analysis/examples/kinetics/t010min.csv \
    mixsense-skills/chem-nmr-mixture-analysis/examples/kinetics/t020min.csv \
    mixsense-skills/chem-nmr-mixture-analysis/examples/kinetics/t030min.csv \
    mixsense-skills/chem-nmr-mixture-analysis/examples/kinetics/t045min.csv \
    mixsense-skills/chem-nmr-mixture-analysis/examples/kinetics/t060min.csv \
    mixsense-skills/chem-nmr-mixture-analysis/examples/kinetics/t090min.csv \
  --times 0 5 10 20 30 45 60 90 \
  --time_unit min \
  --protons 18 18 \
  --names "borneol" "isoborneol" \
  --baseline_correct \
  --output_dir mixsense-skills/chem-nmr-mixture-analysis/examples/results/kinetics
```

Outputs: `kinetics.csv` + `kinetics_plot.png` under `examples/results/kinetics/`.

---

## Constraints

- **Input format**: Two-column numeric CSV or TSV (ppm, intensity), no header required; delimiter auto-detected. `.xy` tab-separated files also accepted.
- **Proton counts**: `--protons` must be provided for accurate molar fractions, matching the order of reference files.
- **Baseline correction**: Recommended (`--baseline-correct`) for digitized or NMRBank spectra. Default flat region is 3.92–4.12 ppm.
- **Wasserstein distance**: Values above ~0.05 indicate a poor fit — check for ppm calibration offsets between mixture and references, missing components, or baseline noise.

- **Magnetstein submodule**: The deconvolution engine requires the `magnetstein` library. Add it as a git submodule in your project root:
  ```bash
  git submodule add https://github.com/BDomzal/magnetstein.git magnetstein
  git submodule update --init --recursive
  ```
  The scripts expect it at `<project_root>/magnetstein/` (three levels above the `scripts/` directory).

- **NMRBank** (for automated reference lookup): Download the NMRBank CSV and set the path:
  ```bash
  export NMRBANK_CSV=/path/to/NMRBank_data_with_SMILES_156621_in_225809.csv
  ```
  Source: [NMRBank repository](https://github.com/liningtonlab/nmrbank)

- **HuggingFace token** (for product prediction): Required to call the ReactionT5 API:
  ```bash
  export HF_TOKEN=your_token   # from https://huggingface.co/settings/tokens
  ```
  Without a token, use `--skip_prediction` to look up references for reactants only.

- **Environment**: Install Python dependencies with:
  ```bash
  uv sync
  ```
  Required packages: `numpy`, `matplotlib`, `pandas`, `rdkit`, `requests`, `pulp`, and `masserstein` (from the magnetstein submodule above).

---

## References

- Górecki, T. et al., "Masserstein: linear resampling of mass spectra by optimal transport", *Rapid Communications in Mass Spectrometry*, 2021. [DOI:10.1002/rcm.9014](https://doi.org/10.1002/rcm.9014)
- Villani, C., "Optimal Transport: Old and New", *Springer*, 2009.
- Dhawan, N. et al., "Synthesis of Isoborneol", *World Journal of Chemical Education*, 2022. [DOI:10.12691/wjce-10-2-1](https://doi.org/10.12691/wjce-10-2-1)
- Sagawa, Y. et al., "ReactionT5: a large-scale pretrained model towards chemical reaction understanding and prediction", *arXiv*, 2023.

---

**Author:** Magdalena Lederbauer
**Contact:** [GitHub @magdalenalederbauer](https://github.com/magdalenalederbauer)
