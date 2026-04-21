# plot-digitizer MCP

Node stdio shim that exposes `digitize_plot` to Claude Desktop
(and any MCP client). Extracts calibrated (x, y) data from a
scientific plot image — NMR, UV-Vis, IR, Raman, chromatograms.

## Setup (2 min, needs Node 18+)

Get a Hugging Face **read token** from https://huggingface.co/settings/tokens
(any read-scoped token works; no write/inference permissions needed).

```bash
git clone https://github.com/jdsanc/MixSense
cd MixSense/.agents/mcp/digitizer
npm install && npm run build

DIGITIZER_BASE_URL=https://jdsan-plot-digitizer-gateway.hf.space \
HF_TOKEN=hf_your_read_token_here \
  npm run print-config
```

Paste the printed snippet into `claude_desktop_config.json`:

| OS      | Path                                                              |
|---------|-------------------------------------------------------------------|
| macOS   | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Windows | `%APPDATA%\Claude\claude_desktop_config.json`                     |
| Linux   | `~/.config/Claude/claude_desktop_config.json`                     |

Restart Claude Desktop. `digitize_plot` appears in the tool list.

## Tool: `digitize_plot`

| Argument        | Type   | Required | Description                                         |
|-----------------|--------|----------|-----------------------------------------------------|
| `image_path`    | string | yes      | Absolute local path to PNG / JPG / TIFF (max 10 MB) |
| `job_plan`      | object | no       | Extraction plan; omit to have Claude auto-generate  |
| `output_format` | string | no       | `json` \| `csv` \| `xy` \| `both` (default `json`)  |

## Notes

- Requests go via a public gateway that authenticates callers by their own
  Hugging Face read token (validated via `whoami-v2`) and proxies to a private
  backend where the digitization source lives. Your token is **not** forwarded
  to the backend.
- Rate limit: 100 requests/day, 10/minute, per HF user.
- Image size cap: 10 MB.
