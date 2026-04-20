# plot-digitizer MCP

Node stdio shim that exposes `digitize_plot` to Claude Desktop
(and any MCP client). Extracts calibrated (x, y) data from a
scientific plot image — NMR, UV-Vis, IR, Raman, chromatograms.

## Setup (2 min, needs Node 18+)

```bash
git clone https://github.com/jdsanc/MixSense
cd MixSense/.agents/mcp/digitizer
npm install && npm run build

DIGITIZER_BASE_URL=https://jdsan-plot-digitizer-gateway.hf.space \
DIGITIZER_API_KEY=<ask-maintainer> \
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

- Requests go via a public gateway that proxies to a private backend
  where the digitization source lives;
  `DIGITIZER_API_KEY` is the only credential you need.
- Image size cap: 10 MB.
