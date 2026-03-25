---
name: chem-nmr-mixture-analysis
description: Quantify components in 1H-NMR spectra of reaction mixtures using blind (PCA/NMF) or supervised (optimal transport / Magnetstein) deconvolution.
category: chemistry
---

# NMR Mixture Analysis

## Goal

Determine the relative abundance of chemical components in a 1H-NMR spectrum of a reaction mixture. Two complementary workflows are supported:

- **Workflow A — Blind Separation (time-series):** Given a sequence of spectra recorded at different time points during a reaction, extract pure-component spectral profiles and their abundance over time using Principal Component Analysis (PCA) for dimensionality estimation and Non-Negative Matrix Factorization (NMF) for source separation. No prior knowledge of component identities is required.

- **Workflow B — Supervised Deconvolution (known references):** Given a mixture spectrum and reference spectra for known components, estimate molar fractions using optimal transport (Wasserstein distance minimization) via the Magnetstein library.

Inputs are two-column `.csv` or `.xy` files with columns `(ppm, intensity)`, as produced by spectrum digitizers or NMR data export tools.

## Instructions

### Workflow A: Blind Separation from Time-Series Spectra

**Use when:** You have multiple NMR spectra recorded over the course of a reaction and want to identify how many distinct species are present and how their abundances evolve — without needing pre-assigned references.

#### Step 1 — Inspect Input Spectra

Before running analysis, visualize the raw spectra to check data quality and identify the ppm region of interest.

```bash
# Env: mixsense
python mixsense-skills/chem-nmr-mixture-analysis/scripts/nmr_plot.py \
  spectra/t0.csv spectra/t1.csv spectra/t2.csv spectra/t3.csv \
  --labels "0 min" "15 min" "30 min" "45 min" \
  --stacked \
  --title "Reaction Time Series" \
  --output results/input_overview.png
```

#### Step 2 — Run PCA + NMF Blind Separation

```bash
# Env: mixsense
python mixsense-skills/chem-nmr-mixture-analysis/scripts/nmr_blind_separation.py \
  spectra/t0.csv spectra/t1.csv spectra/t2.csv spectra/t3.csv \
  --labels "0 min" "15 min" "30 min" "45 min" \
  --ppm_min 0.5 --ppm_max 10.0 \
  --output_dir results/blind_separation
```

This will:
1. Interpolate all spectra to a common ppm grid.
2. Run PCA and save a scree plot (`pca_scree.png`) — inspect this to confirm the estimated component count is chemically reasonable.
3. Run NMF with the estimated (or user-specified) number of components.
4. Save recovered pure-component spectra as `component_0.csv`, `component_1.csv`, ...
5. Save abundance evolution as `abundances.csv`.
6. Save plots: `nmf_components.png`, `nmf_abundances.png`.

To override the auto-detected component count (e.g., force 3 components):

```bash
# Env: mixsense
python mixsense-skills/chem-nmr-mixture-analysis/scripts/nmr_blind_separation.py \
  spectra/t*.csv \
  --n_components 3 \
  --output_dir results/blind_separation_3comp
```

#### Step 3 — Inspect and Interpret Results

```bash
# Env: mixsense
python mixsense-skills/chem-nmr-mixture-analysis/scripts/nmr_plot.py \
  results/blind_separation/component_0.csv \
  results/blind_separation/component_1.csv \
  --labels "NMF Component 0" "NMF Component 1" \
  --title "Recovered Pure Component Spectra" \
  --output results/components_overlay.png
```

Key outputs to check:
- `pca_scree.png`: cumulative variance vs. number of components — confirm the elbow aligns with your chemical expectation.
- `nmf_components.png`: recovered pure spectra — compare peak positions with known reference libraries or literature.
- `nmf_abundances.png`: abundance trends over time — should show monotonic consumption or formation for clean reactions.
- `separation_summary.json`: machine-readable summary of all outputs.

---

### Workflow B: Supervised Deconvolution with Known References

**Use when:** You know which compounds are present (or have reference spectra for them) and want to estimate their molar fractions in a mixture.

#### Step 1 — Inspect Spectra

```bash
# Env: mixsense
python mixsense-skills/chem-nmr-mixture-analysis/scripts/nmr_plot.py \
  mixture.csv ref_compA.csv ref_compB.csv \
  --labels "Mixture" "Component A" "Component B" \
  --title "Mixture vs. References" \
  --output results/mixture_vs_refs.png
```

#### Step 2 — Run Magnetstein Deconvolution

```bash
# Env: mixsense
uv run python app/tool_deconvolve_nmr.py \
  mixture.csv ref_compA.csv ref_compB.csv \
  --protons 16 12 \
  --names "Component A" "Component B" \
  --json
```

Arguments:
- `mixture.csv`: Two-column file (ppm, intensity) for the mixture.
- `ref_compA.csv ref_compB.csv ...`: Reference spectra for each component (same format).
- `--protons 16 12`: Number of 1H protons per molecule for each component (used for normalization). Required for accurate molar fractions.
- `--names "Component A" "Component B"`: Human-readable names for output.
- `--json`: Also print a JSON result line for programmatic parsing.
- `--kappa-mix 0.25` / `--kappa-comp 0.22`: Magnetstein smoothing parameters (rarely need changing).

Output:
```
Estimated proportions:
  Component A    0.623
  Component B    0.377

Wasserstein distance: 0.000142857143

JSON: {"proportions": {"Component A": 0.623, "Component B": 0.377}, "Wasserstein distance": 0.000142857143}
```

The Wasserstein distance is a fit quality metric — lower is better. Values below ~0.001 typically indicate a good fit.

#### Step 3 — Cross-check with Blind Separation (optional)

If the Wasserstein distance is unexpectedly high, or if the proportions seem chemically unreasonable, run Workflow A on the same mixture to check whether additional unaccounted components are present.

---

## Examples

### Example A: Acetylation Reaction Time-Series (Blind Separation)

```bash
# Env: mixsense
python mixsense-skills/chem-nmr-mixture-analysis/scripts/nmr_blind_separation.py \
  examples/acetylation/t0.csv examples/acetylation/t1.csv examples/acetylation/t2.csv \
  --labels "0 min" "30 min" "90 min" \
  --ppm_min 0.5 --ppm_max 9.0 \
  --output_dir examples/acetylation/results
```

Expected: 2 NMF components — one corresponding to starting material (high abundance at t=0, low at t=90min), one to product (low at t=0, high at t=90min).

### Example B: Binary Mixture Quantification (Supervised)

```bash
# Env: mixsense
uv run python app/tool_deconvolve_nmr.py \
  examples/binary_mix/mixture.csv \
  examples/binary_mix/pinene.csv \
  examples/binary_mix/benzyl_benzoate.csv \
  --protons 16 18 \
  --names "alpha-Pinene" "Benzyl benzoate" \
  --json
```

---

## Constraints

- **Input format**: Both workflows require two-column numeric files (ppm, intensity) with no header, or with a header row. Comma-separated (`.csv`) and tab-separated (`.tsv`, `.xy`) are both supported; delimiter is auto-detected.
- **Proton counts**: For supervised deconvolution (Workflow B), `--protons` must be provided for accurate molar fraction estimates. If omitted, proportions are in arbitrary intensity units.
- **Grid interpolation**: Blind separation interpolates all spectra to a common grid. Spectra with very different ppm ranges will be zero-padded outside their range — restrict with `--ppm_min`/`--ppm_max` to the region of interest.
- **NMF non-negativity**: NMF requires non-negative input. Negative intensity values (from baseline drift or phasing artifacts) are clipped to zero before NMF. Pre-process spectra (phase correction, baseline correction) if this causes distortion.
- **Number of spectra**: Blind separation requires at least 3 spectra; more time points give more reliable NMF solutions. The number of extractable components is bounded by the number of input spectra.
- **Environment**: All scripts require the `mixsense` environment. Set up with:
  ```bash
  uv sync
  ```
- **Magnetstein submodule**: The supervised deconvolution tool (`app/tool_deconvolve_nmr.py`) requires the `magnetstein/` submodule to be initialized:
  ```bash
  git submodule update --init --recursive
  ```

## References

- Lee, D. D. & Seung, H. S., "Learning the parts of objects by non-negative matrix factorization", *Nature*, 1999. [DOI:10.1038/44565](https://doi.org/10.1038/44565)
- Górecki, T. et al., "Masserstein: linear resampling of mass spectra by optimal transport", *Rapid Communications in Mass Spectrometry*, 2021. [DOI:10.1002/rcm.9014](https://doi.org/10.1002/rcm.9014)
- Pedregosa, F. et al., "Scikit-learn: Machine Learning in Python", *JMLR*, 2011. [DOI:10.5555/1953048.2078195](https://doi.org/10.5555/1953048.2078195)

---

**Author:** Magdalena Lederbauer
**Contact:** [GitHub @magdalenalederbauer](https://github.com/magdalenalederbauer)
