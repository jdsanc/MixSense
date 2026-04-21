# MixSense

AI-agent framework for quantifying chemical mixtures from crude ¹H NMR spectra.

Describe your reaction and hand over a crude NMR (image or numeric data) and our MixSense agent does the rest for you!

---

## What you get

**Input:** a reaction description + a crude ¹H NMR spectrum
**Output:** mole fractions per component, a fit-quality score, and annotated plots

Worked example shipped in the repo includes reduction of camphor to a borneol/isoborneol mixture:

```
Reaction: NaBH4 reduction of camphor in methanol
Crude spectrum: .agents/skills/nmr-analysis/examples/deconvolution/crude.csv
→ borneol: 0.23, isoborneol: 0.77, WD = 0.04  (good fit)
```

---

## Prerequisites

| Requirement          | Why                                                         | Where to get it                                   |
|----------------------|-------------------------------------------------------------|---------------------------------------------------|
| `conda` / `mamba`    | Single environment for all scripts                          | miniforge / anaconda                              |
| `ANTHROPIC_API_KEY`  | Drive the agent via Claude Code CLI (optional for Desktop)  | console.anthropic.com                             |
| `HF_TOKEN` (read)    | ReactionT5 product prediction + plot-digitizer MCP          | huggingface.co/settings/tokens                    |
| Node 18+             | Only if you want `digitize_plot` MCP (image → data)         | nodejs.org                                        |

Export tokens in your shell:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export HF_TOKEN=hf_...
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/jdsanc/MixSense
cd MixSense
bash conda-envs/mixsense/install.sh
```

Scripts run via:

```bash
conda run -n mixsense python <script> [args]
```

### 2. (Optional) Enable image digitization

If you want to hand the agent a **photo / screenshot** of a spectrum instead of a numeric file, install the `digitize_plot` MCP shim:

```bash
cd .agents/mcp/digitizer
npm install && npm run build

DIGITIZER_BASE_URL=https://jdsan-plot-digitizer-gateway.hf.space \
HF_TOKEN=$HF_TOKEN \
  npm run print-config
```

Paste the printed snippet into your `claude_desktop_config.json` (paths in `.agents/mcp/digitizer/README.md`) and restart Claude Desktop. Skip this step if you only work with numeric `.csv` / `.xy` / `.tsv` files.

---

## Your first run

Start a Claude Code session in the repo root:

```bash
claude           # CLI
# or open the folder in Claude Desktop / VS Code / JetBrains
```

Then paste this prompt verbatim — it runs end-to-end against the bundled example:

> Quantify the mixture in
> `.agents/skills/nmr-analysis/examples/deconvolution/crude.csv`.
> Reaction ran is NaBH4 reduction of camphor in methanol.

Expected: the agent identifies camphor / borneol / isoborneol / methanol, fetches SMILES from PubChem, predicts products via ReactionT5, generates reference spectra with `nmr-predict`, and runs Wasserstein deconvolution. Final plot + mole fractions land in `research/<date>_<slug>/`.

---

## Input formats

| Extension     | Delimiter | Notes                                       |
|---------------|-----------|---------------------------------------------|
| `.csv`        | comma     | Two columns: ppm, intensity. No header.     |
| `.xy`, `.tsv` | tab       | Same shape.                                 |
| `.png`/`.jpg` | —         | Requires `digitize_plot` MCP (step 2 above) |

---

## Choosing a workflow

All three live in `.agents/workflows/`:

| Workflow                           | Use when                                                                 |
|------------------------------------|--------------------------------------------------------------------------|
| `reaction-to-nmr-quantification`   | Single crude spectrum (numeric). Want mole fractions at one time point. |
| `image-to-nmr-analysis`            | Same as above, but input is an **image** of the spectrum.                |
| `nmr-reaction-kinetics`            | Multiple time-point spectra. Want mole-fraction-vs-time curves.          |

Each workflow composes skills from `.agents/skills/`:

```bash
# Discover skills
grep -r "^description:" .agents/skills/*/SKILL.md
```

---

## Reading the output

The agent reports a **Wasserstein distance (WD)** per fit:

| WD range   | Meaning                                                 |
|------------|---------------------------------------------------------|
| `< 0.05`   | Good fit. Trust mole fractions.                         |
| `0.05–0.15`| Acceptable. Inspect overlay plot before trusting.       |
| `> 0.15`   | Poor. Likely missing component or ppm-referencing drift.|

If WD is high, re-check that all NMR-visible species (solvent, reagents, byproducts) are in the component list.

---

## Troubleshooting

- **`HF_TOKEN` invalid** — token must have read scope; regenerate at huggingface.co/settings/tokens.
- **Digitizer rate-limited** — 100 req/day, 10/min per HF user.
- **ReactionT5 returns no products** — agent will fall back to asking you; confirm products from chemistry knowledge.
- **Peaks don't align** — usually a ppm-referencing offset in the crude spectrum; the agent will flag it.

---

See `CLAUDE.md` for full architecture, skill standards, and coding conventions.
