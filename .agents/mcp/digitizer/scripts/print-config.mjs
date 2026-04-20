#!/usr/bin/env node
// Prints the claude_desktop_config.json snippet for this machine.
//
//   node scripts/print-config.mjs \
//       --base-url https://jdsan-plot-digitizer.hf.space \
//       --api-key $DIGITIZER_API_KEY
//
// Or via env:
//   DIGITIZER_BASE_URL=... DIGITIZER_API_KEY=... node scripts/print-config.mjs

import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { existsSync } from "node:fs";

const here = dirname(fileURLToPath(import.meta.url));
const distPath = resolve(here, "..", "dist", "index.js");

if (!existsSync(distPath)) {
  console.error(`error: dist/index.js not found at ${distPath}`);
  console.error("       run 'npm install && npm run build' first.");
  process.exit(1);
}

const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, arr) => {
    if (a.startsWith("--")) acc.push([a.slice(2), arr[i + 1]]);
    return acc;
  }, [])
);

const baseUrl = args["base-url"] ?? process.env.DIGITIZER_BASE_URL;
const apiKey = args["api-key"] ?? process.env.DIGITIZER_API_KEY;

if (!baseUrl || !apiKey) {
  console.error("error: need --base-url and --api-key (or DIGITIZER_BASE_URL / DIGITIZER_API_KEY env vars).");
  process.exit(1);
}

const snippet = {
  mcpServers: {
    "plot-digitizer": {
      command: "node",
      args: [distPath],
      env: {
        DIGITIZER_BASE_URL: baseUrl,
        DIGITIZER_API_KEY: apiKey,
      },
    },
  },
};

console.log("\n# Paste into claude_desktop_config.json");
console.log("# macOS:   ~/Library/Application Support/Claude/claude_desktop_config.json");
console.log("# Windows: %APPDATA%\\Claude\\claude_desktop_config.json");
console.log("# Linux:   ~/.config/Claude/claude_desktop_config.json\n");
console.log(JSON.stringify(snippet, null, 2));
console.log("\n# If the file already has mcpServers, merge 'plot-digitizer' into it.\n");
