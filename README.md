# MixSense

AI-agent framework for quantifying chemical mixtures from crude ¹H NMR spectra.

## Setup

### 1. Clone

```bash
git clone <repo-url>
cd MixSense
```

### 2. Environment

```bash
bash conda-envs/mixsense/install.sh
```

Run any script with:

```bash
conda run -n mixsense python <script> [args]
```

---

## Using with Claude Code

MixSense ships with an agent framework (`.agents/`) designed for Claude Code.

### Desktop App (claude.ai)

1. Open [claude.ai/code](https://claude.ai/code)
2. Connect to this repo via **Open Folder**
3. Claude reads `CLAUDE.md` automatically on session start

### VS Code / JetBrains IDE

1. Install the **Claude Code** extension from the marketplace
2. Open the MixSense folder
3. Use `Cmd/Ctrl+Shift+P` → **Claude Code: Start Session**

### CLI (terminal / coding agents)

```bash
# Install
npm install -g @anthropic-ai/claude-code

# Run in repo root
cd MixSense
claude
```

For headless / agent use:

```bash
claude --print "describe the nmr-predict skill"
```

> Requires `ANTHROPIC_API_KEY` in your environment.

---

## Quick Start

```bash
# Discover available skills
grep -r "^description:" .agents/skills/*/SKILL.md

# See end-to-end workflows
ls .agents/workflows/
```

See `CLAUDE.md` for full architecture and coding standards.
