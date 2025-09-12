# LLMHackathon



# 🧪 NMR Mixture Deconvolution

**Script:** `deconvolve_nmr.py`  
**Goal:** Deconvolve a 1D NMR mixture spectrum into contributions from known component spectra using [Masserstein](https://github.com/BDomzal/masserstein) optimal transport and the Gurobi linear programming solver (via PuLP).

---

## 📥 Inputs

**Required:**
- **Mixture spectrum** (`mixture`):  
  CSV or TSV file with two columns:  
  - Column 1: ppm (chemical shift)  
  - Column 2: intensity  

- **Component spectra** (`components`):  
  One or more CSV/TSV files with the same format (ppm, intensity).

**Optional:**
- `--protons`: Number of protons per component (for intensity normalization; must match number of components)
- `--names`: Custom names for components (for output table and JSON)
- `--mnova`: If files are Mnova TSV exports

---

## ⚙️ Main Arguments

| Flag                | Description                                                     | Default          |
|---------------------|----------------------------------------------------------------|------------------|
| `--threads`          | Number of BLAS / solver threads                                | 8                 |
| `--time-limit`       | Gurobi solver time limit (seconds)                             | 300               |
| `--method`            | LP solve method: `auto`, `primal`, `dual`, `barrier`            | dual               |
| `--presolve`           | Gurobi presolve level (0–2)                                    | 2                   |
| `--numeric-focus`       | NumericFocus (0–3, higher = more numerical stability)         | 1                    |
| `--skip-crossover`       | Skip crossover when using barrier method                     | off                  |
| `--license-file`         | Path to `gurobi.lic` license file                            | env by default         |
| `--json`                   | Print output also in JSON                                  | off                       |
| `--quiet`                     | Reduce solver verbosity                                 | off                          |

---

## ▶️ Example Usage

**With preprocessed CSV spectra:**
```bash
python scripts/deconvolve_nmr.py \
  magnetstein/examples/preprocessed_mix.csv \
  magnetstein/examples/preprocessed_comp0.csv \
  magnetstein/examples/preprocessed_comp1.csv \
  --protons 16 12 \
  --names Pinene "Benzyl benzoate" \
  --threads 8 \
  --method dual \
  --presolve 2 \
  --time-limit 300 \
  --json \
  --license-file /home/luciavl/.gurobi/gurobi.lic
