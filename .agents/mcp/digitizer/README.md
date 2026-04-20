# plot-digitizer MCP

MCP shim that exposes the `digitize_plot` tool for Claude Desktop (and any
MCP-compatible client). Extracts calibrated (x, y) numeric data from a
scientific plot or spectrum image (NMR, UV-Vis, Raman, IR, chromatograms, etc.).

Zero business logic lives here — the shim is a thin client that sends the
image to a hosted API. The digitization pipeline stays server-side.

## Architecture

```
┌─────────────────────────┐    stdio   ┌──────────────────────────┐
│ Claude Desktop / Cursor │──────────▶│ plot-digitizer MCP shim   │
│ (MCP client)            │            │ (this Node process)       │
└─────────────────────────┘            └────────────┬─────────────┘
                                                    │ HTTPS
                                                    │ X-API-Key: <DIGITIZER_API_KEY>
                                                    ▼
                                       ┌──────────────────────────┐
                                       │ plot-digitizer-gateway   │  public
                                       │ (HF Space, thin proxy)   │
                                       └────────────┬─────────────┘
                                                    │ HTTPS
                                                    │ Authorization: Bearer <HF_TOKEN>
                                                    │ X-API-Key: <...>
                                                    ▼
                                       ┌──────────────────────────┐
                                       │ plot-digitizer backend   │  private
                                       │ (HF Space, Docker, CPU)  │  (source hidden)
                                       └──────────────────────────┘
```

Claude generates the **job plan** client-side (via MCP sampling), so the
backend is CPU-only. The gateway holds the HF token as a Space Secret,
letting end users reach the private backend with just `DIGITIZER_API_KEY` —
no HF account needed.

---

## End-user setup (Claude Desktop)

Three steps. Takes ~2 minutes once you have Node.js.

### 1. Install + build

Needs Node.js 18+.

```bash
git clone https://github.com/<owner>/MixSense
cd MixSense/.agents/mcp/digitizer
npm install
npm run build
```

### 2. Get the API key

Ask the skill maintainer for the `DIGITIZER_API_KEY`. The gateway URL is
fixed at `https://jdsan-plot-digitizer-gateway.hf.space` (public).

### 3. Generate + install the config snippet

```bash
DIGITIZER_BASE_URL=https://jdsan-plot-digitizer-gateway.hf.space \
DIGITIZER_API_KEY=<key> \
  npm run print-config
```

This prints a ready-to-paste JSON snippet. Copy it into:

| OS      | Path                                                             |
|---------|------------------------------------------------------------------|
| macOS   | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json`                    |
| Linux   | `~/.config/Claude/claude_desktop_config.json`                    |

If the file already has an `mcpServers` key, merge `plot-digitizer` into it.

Restart Claude Desktop. `digitize_plot` appears in the tool list.

---

## Tool: `digitize_plot`

| Argument        | Type    | Required | Description                                             |
|-----------------|---------|----------|---------------------------------------------------------|
| `image_path`    | string  | yes      | Absolute local path to PNG / JPG / TIFF (max 10 MB)     |
| `job_plan`      | object  | no       | Extraction plan. Omit to have Claude auto-generate one. |
| `output_format` | string  | no       | `json` \| `csv` \| `xy` \| `both` (default `json`)      |

The shim enforces a file-extension allowlist (`.png`, `.jpg`, `.jpeg`,
`.tif`, `.tiff`), rejects symlinks/directories, and refuses >10 MB.

---

## API contract

```
POST /v1/digitize
X-API-Key: <DIGITIZER_API_KEY>
Content-Type: application/json

{
  "image_b64": "<base64>",
  "image_mime": "image/png",
  "image_filename": "spectrum.png",
  "output_format": "json",
  "job_plan": { ... }
}

→ 200 { job_id, curves: [...], warnings }
→ 401  AUTH_FAILED
→ 429  rate limited (10/min, 100/day per key)
→ 422  invalid job_plan
→ 500  pipeline error
```

`GET /health` → `{"status":"ok"}` — no auth.

---

## Security

- **`X-API-Key` auth** checked on the backend via `hmac.compare_digest`.
- **Rate limit** 10/min, 100/day keyed by `X-API-Key` (per-user bucket).
- **Image size cap** 10 MB raw; rejected both client-side (shim), mid-tier (gateway 16 MB body cap), and server-side (pydantic).
- **File allowlist** in the shim — only `.png/.jpg/.jpeg/.tif/.tiff`, regular files (no symlinks, no dirs).
- **Private backend Space** — source code (`spectra/`) is not publicly downloadable.
- **HF Secrets** hold `DIGITIZER_API_KEY` (backend) and `HF_TOKEN` / `BACKEND_URL` (gateway) — never in git or Space logs.
- **Fine-grained HF token** — gateway uses a token with *read-only* access to only the backend Space, not the full account.
- **No shell exec**, no filesystem writes outside cwd.

### Key rotation

Regenerate:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Update the secret on the Space (`Settings → Variables and secrets`), then hand
the new key to users. The Space restarts automatically.

### Hardening options (optional)

- **Cloudflare Access** (zero-trust email gate) in front of the Space — requires
  a Cloudflare Worker proxying the HF URL; out of scope for default setup.
- **Per-user keys** — extend `api/auth.py` to look up keys in a small JSON
  file; keeps the rate-limit bucket per-user.

---

## Local dev

```bash
npm install
npm run build
DIGITIZER_BASE_URL=http://127.0.0.1:7860 \
DIGITIZER_API_KEY=dev-key \
  node dist/index.js < test-input.jsonl
```

The shim talks stdio JSON-RPC — point an MCP client at `dist/index.js`.

---

## Deploying the API (operator only)

Server code lives in `.local/digitizer/` (gitignored). To deploy to Hugging
Face Spaces:

```bash
cd .local/digitizer
huggingface-cli login                    # one-time
DIGITIZER_API_KEY=<strong-key> \
  bash deploy/hf-deploy.sh
```

See `.local/digitizer/README.md` for full operator docs.
