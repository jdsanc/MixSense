---
name: chem-nmr-mixture-analysis
description: Quantify components in 1H-NMR spectra of reaction mixtures via supervised Wasserstein deconvolution against reference spectra — either for a single time point or across a reaction time series to extract kinetics.
category: chemistry
---

# NMR Mixture Analysis

## Goal

Determine the molar composition of a 1H-NMR mixture spectrum by deconvolving it against reference spectra for each known component. Requires:

1. **Reference spectra** for each component — from a database (NMRBank), simulation, or measured pure-compound CSV files.
2. **Crude spectrum** at one or more time points to be deconvolved.

Two use cases, both supervised:

- **Single time point**: one crude spectrum + references → estimated mole fractions + Wasserstein distance (fit quality).
- **Reaction kinetics (time series)**: crude spectra recorded at multiple times + same references → mole fraction vs time table and kinetics plot.

All input files are two-column `.csv` or `.xy` (ppm, intensity), as produced by spectrum digitizers or NMR software (e.g. Mnova).

---

## Instructions

### Step 1 — Prepare reference spectra

References must be pure-component spectra in two-column CSV format (ppm, intensity). Three sources are supported:

| Source | How |
|--------|-----|
| Measured pure compound | Export from Mnova / NMR instrument software |
| Digitized from literature | Use a spectrum digitizer, save as CSV |
| NMRBank database | Look up by SMILES via `app/tools_nmrbank.py` |

> **Tip:** Reference spectra should be recorded under the same conditions (solvent, field strength) as the mixture. Mismatches increase the Wasserstein distance.

### Step 2 — Inspect spectra visually

Before deconvolving, overlay all spectra to spot calibration offsets or baseline drift:

```bash
# Env: mixsense
uv run python mixsense-skills/chem-nmr-mixture-analysis/scripts/nmr_plot.py \
  crude.csv ref_a.csv ref_b.csv \
  --labels "Mixture" "Component A" "Component B" \
  --title "Mixture vs. References" \
  --output results/spectra_overview.png
```

### Step 3a — Single time point: supervised deconvolution

```bash
# Env: mixsense
uv run python app/tool_deconvolve_nmr.py \
  crude.csv ref_a.csv ref_b.csv \
  --protons 18 18 \
  --names "borneol" "isoborneol" \
  --baseline-correct \
  --plot results/deconvolution_result.png \
  --json
```

Key arguments:
- `--protons 18 18`: number of 1H protons per molecule for each component. Required for accurate molar fractions. Must match the order of reference files.
- `--names`: human-readable labels for output and plot.
- `--baseline-correct`: subtracts the median intensity of a flat ppm region (default 3.92–4.12 ppm) from all spectra before deconvolution. **Recommended for digitized spectra** where the baseline may be non-zero. Override with `--baseline-flat-min` / `--baseline-flat-max` if needed.
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

### Step 3b — Time series: reaction kinetics

When crude spectra are available at multiple time points, run deconvolution at each point and collect results into a kinetics table and plot:

```bash
# Env: mixsense
uv run python mixsense-skills/chem-nmr-mixture-analysis/scripts/nmr_kinetics.py \
  --refs ref_a.csv ref_b.csv \
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

### Example A — Single time point: camphor NaBH₄ reduction

Deconvolve the crude reaction mixture against borneol and isoborneol reference spectra (digitized from Dhawan et al., 2022).

```bash
# Env: mixsense
uv run python app/tool_deconvolve_nmr.py \
  mixsense-skills/chem-nmr-mixture-analysis/examples/deconvolution/crude.csv \
  mixsense-skills/chem-nmr-mixture-analysis/examples/deconvolution/borneol.csv \
  mixsense-skills/chem-nmr-mixture-analysis/examples/deconvolution/isoborneol.csv \
  --protons 18 18 \
  --names "borneol" "isoborneol" \
  --baseline-correct \
  --plot mixsense-skills/chem-nmr-mixture-analysis/examples/results/deconvolution_result.png \
  --json
```

Expected: isoborneol is the major product (~80%), consistent with the known stereoselectivity of NaBH₄ reduction of camphor (literature: ~1:4 borneol:isoborneol).

### Example B — Reaction kinetics: simulated camphor reduction time series

Synthetic crude spectra at 8 time points generated by linearly combining the borneol and isoborneol references with a 1st-order kinetic profile (borneol → isoborneol, k = 0.04 min⁻¹, asymptote 20:80). Ground truth stored in `examples/kinetics/`.

```bash
# Env: mixsense
uv run python mixsense-skills/chem-nmr-mixture-analysis/scripts/nmr_kinetics.py \
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
- **Baseline correction**: Recommended (`--baseline-correct`) for digitized spectra. Default flat region is 3.92–4.12 ppm; override with `--baseline-flat-min` / `--baseline-flat-max`.
- **Wasserstein distance**: Values above ~0.05 indicate a poor fit — check for ppm calibration offsets between mixture and references, missing components, or baseline noise.
- **Magnetstein submodule**: Requires the `magnetstein/` submodule to be initialised:
  ```bash
  git submodule update --init --recursive
  ```
- **Environment**: Set up with:
  ```bash
  uv sync
  ```

---

## References

- Górecki, T. et al., "Masserstein: linear resampling of mass spectra by optimal transport", *Rapid Communications in Mass Spectrometry*, 2021. [DOI:10.1002/rcm.9014](https://doi.org/10.1002/rcm.9014)
- Villani, C., "Optimal Transport: Old and New", *Springer*, 2009.
- Dhawan, N. et al., "Synthesis of Isoborneol", *World Journal of Chemical Education*, 2022. [DOI:10.12691/wjce-10-2-1](https://doi.org/10.12691/wjce-10-2-1)

---

**Author:** Magdalena Lederbauer
**Contact:** [GitHub @magdalenalederbauer](https://github.com/magdalenalederbauer)
