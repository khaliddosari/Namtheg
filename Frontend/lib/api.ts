// Every request goes through the same-origin /api/backend proxy defined in
// next.config.ts. The browser never contacts the backend directly, so the backend
// URL is never baked into the client bundle and there are no CORS concerns — a
// down/suspended backend surfaces as a normal HTTP error instead of an opaque
// "Failed to fetch". Set BACKEND_URL in the Vercel project's Environment Variables
// to your Render backend's public URL (see Docs/DEPLOYMENT.md). In local dev the
// proxy forwards to http://localhost:8000 (next.config.ts default).
function resolveBase(): string {
  // Browser — the only real caller, since every page using this is a client component.
  if (typeof window !== "undefined") return "/api/backend";
  // Defensive SSR/build fallback: talk to the backend directly if ever called server-side.
  const raw = process.env.BACKEND_URL ?? "http://localhost:8000";
  return /^https?:\/\//i.test(raw) ? raw.replace(/\/+$/, "") : `https://${raw}`;
}

const BASE = resolveBase();


export interface UploadResponse {
  run_id: string;
  filename: string;
  columns: string[];
  preview: Record<string, unknown>[];
}

export interface RunStatus {
  run_id: string;
  status: "uploaded" | "queued" | "running" | "succeeded" | "failed";
  target?: string;
  error?: string;
}

export interface ModelScore {
  name: string;
  test_accuracy?: number;
  test_r2?: number;
  cv_mean: number;
  cv_std: number;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface TuningTrial {
  trial: number;
  parameters: string;
  score: number | null; // null when the trial failed
  result: string;
}

export interface ResultExtra {
  train_accuracy?: number;
  train_r2?: number;
  overfit_gap?: number;
  f1_macro?: number;
  cv_accuracy_mean?: number;
  cv_r2_mean?: number;
  rmse?: number;
  mae?: number;
  n_classes?: number;
  test_size?: number;
  all_models?: ModelScore[];
  top_features?: FeatureImportance[];
  tuning_trials?: TuningTrial[];
  train_score?: number; // selection metric on the training split
  test_score?: number; // selection metric on the held-out test split
  test_metrics?: Record<string, number>;
  hardware?: { gpu?: string };
}

export interface RunResult {
  run_id: string;
  status: string;
  target?: string;
  problem_type?: "regression" | "classification";
  accuracy_score?: number;
  score_metric?: string;
  plot_path?: string;
  justification?: string;
  model_name?: string;
  extra?: ResultExtra;
  error?: string;
  downloads?: DownloadKind[];
}

export type DownloadKind = "cleaned_csv" | "model";

// A plain link: the backend streams the file, or redirects to a short-lived
// R2 URL when R2 storage is configured.
export function downloadUrl(runId: string, kind: DownloadKind): string {
  return `${BASE}/runs/${runId}/download/${kind}`;
}

export async function uploadCSV(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function startRun(runId: string, target: string): Promise<RunStatus> {
  const res = await fetch(`${BASE}/runs/${runId}/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ target }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getStatus(runId: string): Promise<RunStatus> {
  const res = await fetch(`${BASE}/runs/${runId}/status`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getResult(runId: string): Promise<RunResult> {
  const res = await fetch(`${BASE}/runs/${runId}/result`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function plotUrl(runId: string): string {
  return `${BASE}/runs/${runId}/plot`;
}

export interface Deployment {
  run_id: string;
  status: "not_deployed" | "deploying" | "succeeded" | "failed";
  app_name?: string;
  predict_url?: string | null;
  schema_url?: string | null;
  all_urls?: string[];
  error?: string | null;
  started_at?: string;
  finished_at?: string;
  elapsed_seconds?: number;
}

export async function startDeploy(runId: string): Promise<Deployment> {
  const res = await fetch(`${BASE}/runs/${runId}/deploy`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getDeployment(runId: string): Promise<Deployment> {
  const res = await fetch(`${BASE}/runs/${runId}/deployment`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface ModelSchema {
  run_id: string;
  model_name?: string;
  problem_type?: "regression" | "classification";
  feature_cols: string[];
  class_labels?: string[] | null;
  sample: Record<string, number | string | null>;
}

export interface PredictionResponse {
  predictions?: (number | string)[];
  predicted_labels?: (string | null)[];
  probabilities?: number[][];
  class_labels?: string[];
  model?: string;
  error?: string;
}

export async function getModelSchema(runId: string): Promise<ModelSchema> {
  const res = await fetch(`${BASE}/runs/${runId}/model_schema`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function predict(
  runId: string,
  payload: { features: Record<string, number | string | null> } | { rows: (number | string | null)[][] },
): Promise<PredictionResponse> {
  const res = await fetch(`${BASE}/runs/${runId}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface PreviewData {
  run_id: string;
  filename?: string;
  columns: string[];
  n_columns: number;
  preview: Record<string, unknown>[];
}

export async function getPreview(runId: string): Promise<PreviewData> {
  const res = await fetch(`${BASE}/runs/${runId}/preview`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export interface Diagnostics {
  cpu: number;
  gpu: number;
  ram: number;
  ram_total: number;
  speed: number;
}

export async function getDiagnostics(runId: string): Promise<Diagnostics> {
  const res = await fetch(`${BASE}/runs/${runId}/diagnostics`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


