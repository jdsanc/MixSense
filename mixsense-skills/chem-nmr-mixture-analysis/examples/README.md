# Examples: chem-nmr-mixture-analysis

This directory contains reference examples for the NMR mixture analysis skill.

## Example A — Acetylation Reaction Time-Series (Blind Separation)

**Goal:** Recover pure-component NMR spectra from a time-series of mixture spectra recorded during an acetylation reaction, without prior knowledge of the components.

**Inputs:**
- `acetylation/t0.csv` — spectrum at t = 0 min (pure starting material)
- `acetylation/t1.csv` — spectrum at t = 30 min (mixture)
- `acetylation/t2.csv` — spectrum at t = 90 min (predominantly product)

Each file: two-column CSV (ppm, intensity), ppm range 0–10 ppm, 1000 data points.

**Expected outputs** (in `acetylation/results/`):
- `pca_scree.png` — sharp elbow at 2 components
- `component_0.csv`, `component_1.csv` — recovered spectra matching starting material and product
- `nmf_abundances.png` — starting material decreasing, product increasing monotonically

**Run:**
```bash
# Env: mixsense
python mixsense-skills/chem-nmr-mixture-analysis/scripts/nmr_blind_separation.py \
  examples/acetylation/t0.csv examples/acetylation/t1.csv examples/acetylation/t2.csv \
  --labels "0 min" "30 min" "90 min" \
  --ppm_min 0.5 --ppm_max 9.0 \
  --output_dir examples/acetylation/results
```

## Example B — Binary Mixture Quantification (Supervised)

**Goal:** Estimate molar fractions of α-Pinene and Benzyl benzoate in a known binary mixture.

**Inputs:**
- `binary_mix/mixture.csv` — mixture spectrum
- `binary_mix/pinene.csv` — α-Pinene reference (16 protons)
- `binary_mix/benzyl_benzoate.csv` — Benzyl benzoate reference (12 protons)

**Expected output:**
```
Estimated proportions:
  alpha-Pinene      0.55
  Benzyl benzoate   0.45
Wasserstein distance: ~0.0001
```

**Run:**
```bash
# Env: mixsense
uv run python app/tool_deconvolve_nmr.py \
  examples/binary_mix/mixture.csv \
  examples/binary_mix/pinene.csv \
  examples/binary_mix/benzyl_benzoate.csv \
  --protons 16 12 \
  --names "alpha-Pinene" "Benzyl benzoate" \
  --json
```
