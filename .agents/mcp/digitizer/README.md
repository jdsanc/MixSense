# pendar-digitizer MCP

Thin MCP shim that exposes the `digitize_plot` tool. Zero business logic — calls the workstation HTTPS API.

## What runs where

```
Your machine (Claude / agent)
  └─ MCP shim (this)  →  HTTPS  →  Workstation (FastAPI + Gemma + digitizer source)
                                    ├── Cloudflare Tunnel
                                    ├── uvicorn api.main:app --port 8000
                                    ├── Gemma (VLM for auto job-plan)
                                    └── digitizer source (never leaves workstation)
```

## Workstation setup

```bash
# 1. Install cloudflared
curl -L https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main" | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install cloudflared

# 2. Start API (inside mixsense env)
conda activate mixsense
uvicorn api.main:app --host 127.0.0.1 --port 8000

# 3. Start tunnel (prints your permanent HTTPS URL)
cloudflared tunnel --url http://localhost:8000
```

For persistence across reboots, create two systemd services (uvicorn + cloudflared).

## Model weights

The workstation API uses a local Gemma model (via Ollama or HuggingFace Transformers) to auto-generate job plans when `job_plan` is not provided by the caller. Weights are cached at `~/.cache/huggingface/` or `~/.ollama/` and never leave the workstation.

## Client setup

```bash
cp .env.example .env
# fill in DIGITIZER_BASE_URL and DIGITIZER_API_KEY (get from repo owner)

npm install
npm run build
```

## Claude Desktop / Cursor config

```json
{
  "mcpServers": {
    "pendar-digitizer": {
      "command": "node",
      "args": ["/absolute/path/to/.agents/mcp/digitizer/dist/index.js"],
      "env": {
        "DIGITIZER_BASE_URL": "https://your-tunnel.trycloudflare.com",
        "DIGITIZER_API_KEY": "your-secret-key"
      }
    }
  }
}
```

## Tool: `digitize_plot`

| Argument | Type | Required | Description |
|---|---|---|---|
| `image_path` | string | yes | Absolute path to PNG/JPG/TIFF (max 10 MB) |
| `job_plan` | object | no | VLM extraction plan; omit for auto-planning |
| `output_format` | string | no | `json` \| `csv` \| `xy` \| `both` (default: `json`) |

Rate limits (enforced server-side): **10 req/min, 100 req/day**.

## API contract

```
POST /v1/digitize
Authorization: Bearer <DIGITIZER_API_KEY>
Content-Type: application/json

{
  "image_b64": "<base64>",
  "image_mime": "image/png",
  "image_filename": "spectrum.png",
  "output_format": "json",
  "job_plan": { ... }   // optional
}

→ 200 { job_id, curves: [ { label, x_data, y_data, quality_score, n_points, x_coverage, error } ], warnings }
→ 401  AUTH_FAILED
→ 429  rate limited
→ 500  pipeline error

GET /health  →  { status: "ok" }  (no auth)
```
