import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import { readFileSync, statSync } from "fs";
import { resolve, extname } from "path";
import { DIGITIZE_TOOL, JOB_PLAN_PROMPT } from "./tools.js";
import { digitizePlot } from "./client.js";

const ALLOWED_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".tif", ".tiff"]);

function safeReadImage(imagePath: string): Buffer {
  const resolved = resolve(imagePath);

  // Must be an allowed image extension
  if (!ALLOWED_EXTENSIONS.has(extname(resolved).toLowerCase())) {
    throw new Error(`Rejected: ${extname(resolved)} is not an allowed image type`);
  }

  // Must be a regular file (no symlinks to sensitive paths, no dirs)
  const stat = statSync(resolved);
  if (!stat.isFile()) {
    throw new Error("Rejected: path is not a regular file");
  }

  return readFileSync(resolved);
}

const server = new Server(
  { name: "plot-digitizer", version: "1.1.0" },
  { capabilities: { tools: {} } }
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [DIGITIZE_TOOL],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  if (request.params.name !== "digitize_plot") {
    throw new Error(`Unknown tool: ${request.params.name}`);
  }

  const args = request.params.arguments as {
    image_path: string;
    job_plan?: object;
    output_format?: string;
  };

  const imageBytes = safeReadImage(args.image_path);
  const imageB64 = imageBytes.toString("base64");
  const imageMime = args.image_path.match(/\.png$/i)
    ? "image/png"
    : args.image_path.match(/\.tiff?$/i)
      ? "image/tiff"
      : "image/jpeg";

  let jobPlan: object;

  if (args.job_plan !== undefined) {
    jobPlan = args.job_plan;
  } else {
    const sampling = await server.createMessage({
      messages: [
        {
          role: "user",
          content: { type: "image", data: imageB64, mimeType: imageMime },
        },
        {
          role: "user",
          content: { type: "text", text: JOB_PLAN_PROMPT },
        },
      ],
      systemPrompt:
        "You are a scientific plot analysis assistant. " +
        "Return only valid JSON — no prose, no markdown fences.",
      maxTokens: 1024,
      includeContext: "none",
    });

    const raw =
      sampling.content.type === "text" ? sampling.content.text.trim() : "";
    const cleaned = raw.replace(/^```(?:json)?\s*/m, "").replace(/\s*```$/m, "");
    jobPlan = JSON.parse(cleaned);
  }

  const result = await digitizePlot(imageB64, imageMime, jobPlan, args.output_format ?? "json");

  const curveSummary = result.curves
    .map((c) =>
      c.error
        ? `  [${c.label}] ERROR: ${c.error}`
        : `  [${c.label}] ${c.n_points} pts, quality=${c.quality_score.toFixed(3)}, coverage=${c.x_coverage.toFixed(3)}`
    )
    .join("\n");

  const header =
    `Digitized ${result.curves.length} curve(s) — job_id=${result.job_id}\n${curveSummary}` +
    (result.warnings.length ? `\nWarnings: ${result.warnings.join("; ")}` : "");

  return {
    content: [
      { type: "text", text: header },
      { type: "text", text: JSON.stringify(result, null, 2) },
    ],
  };
});

const transport = new StdioServerTransport();
await server.connect(transport);
