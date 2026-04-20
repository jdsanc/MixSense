import type { Tool } from "@modelcontextprotocol/sdk/types.js";

export const JOB_PLAN_PROMPT = `Analyze this scientific plot image and return ONLY a JSON job plan.

Schema:
{
  "panels": [
    {
      "panel_id": "panel_0",
      "bounding_box": {"x_min": <int>, "y_min": <int>, "x_max": <int>, "y_max": <int>},
      "x_range": [<float>, <float>],
      "y_range": [<float>, <float>],
      "x_label": "<string>",
      "y_label": "<string>",
      "x_scale": "linear" | "log",
      "y_scale": "linear" | "log",
      "curves": [
        {"label": "<string>", "color": "<hex e.g. #1f77b4>", "algorithm": "xstep" | "averaging"}
      ]
    }
  ]
}

Rules:
- bounding_box: pixel coords of the inner plot area (exclude axis labels and titles)
- x_range / y_range: numeric data range read from axis tick labels, [min, max]
- algorithm: "xstep" for smooth continuous curves, "averaging" for noisy/thick traces
- One curve entry per distinct color
- Multiple sub-panels → one object per panel
- Raw JSON only, no markdown.`;

export const DIGITIZE_TOOL: Tool = {
  name: "digitize_plot",
  description:
    "Extract calibrated X-Y numeric data from a scientific plot or spectrum image " +
    "(NMR, UV-Vis, Raman, IR, chromatogram, generic X-Y). " +
    "If job_plan is omitted, Claude analyzes the image automatically via MCP sampling. " +
    "Returns x_data / y_data arrays in calibrated axis units plus quality diagnostics.",
  inputSchema: {
    type: "object",
    properties: {
      image_path: {
        type: "string",
        description: "Absolute local path to the plot image (PNG, JPG, TIFF, max 10 MB)",
      },
      job_plan: {
        type: "object",
        description:
          "Optional extraction plan. If omitted, Claude auto-generates one. " +
          "Structure: { panels: [ { panel_id, bounding_box: {x_min,y_min,x_max,y_max}, " +
          "x_range: [min,max], y_range: [min,max], x_scale, y_scale, " +
          "curves: [ { label, color, algorithm } ] } ] }",
      },
      output_format: {
        type: "string",
        enum: ["json", "csv", "xy", "both"],
        description: "Output format for curve data. Default: json",
      },
    },
    required: ["image_path"],
  },
};
