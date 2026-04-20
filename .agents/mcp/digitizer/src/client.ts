import { basename } from "path";

interface Config {
  baseUrl: string;
  apiKey: string;
}

export interface CurveResult {
  label: string;
  panel_id: string;
  x_data: number[];
  y_data: number[];
  csv_data?: string;
  xy_data?: string;
  quality_score: number;
  n_points: number;
  x_coverage: number;
  error: string | null;
}

export interface DigitizerResponse {
  job_id: string;
  curves: CurveResult[];
  warnings: string[];
}

function getConfig(): Config {
  const baseUrl = process.env.DIGITIZER_BASE_URL;
  const apiKey = process.env.DIGITIZER_API_KEY;
  if (!baseUrl) throw new Error("DIGITIZER_BASE_URL env var not set");
  if (!apiKey) throw new Error("DIGITIZER_API_KEY env var not set");
  return { baseUrl: baseUrl.replace(/\/$/, ""), apiKey };
}

export async function digitizePlot(
  imageB64: string,
  imageMime: string,
  jobPlan: object,
  outputFormat = "json"
): Promise<DigitizerResponse> {
  const cfg = getConfig();

  if ((imageB64.length * 3) / 4 > 10 * 1024 * 1024) {
    throw new Error("Image exceeds 10 MB limit");
  }

  const body = {
    image_b64: imageB64,
    image_mime: imageMime,
    output_format: outputFormat,
    job_plan: jobPlan,
  };

  const resp = await fetch(`${cfg.baseUrl}/v1/digitize`, {
    method: "POST",
    headers: {
      "X-API-Key": cfg.apiKey,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (resp.status === 401) throw new Error("AUTH_FAILED — check DIGITIZER_API_KEY");
  if (resp.status === 429) {
    const data = (await resp.json().catch(() => ({}))) as { retry_after_seconds?: number };
    throw new Error(`RATE_LIMIT_EXCEEDED — retry after ${data.retry_after_seconds ?? 60}s`);
  }
  if (!resp.ok) {
    const data = (await resp.json().catch(() => ({}))) as { error?: string; detail?: string };
    throw new Error(`API_ERROR ${resp.status}: ${data.error ?? data.detail ?? resp.statusText}`);
  }

  return resp.json() as Promise<DigitizerResponse>;
}
