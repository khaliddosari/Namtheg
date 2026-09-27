// Every request goes through the same-origin /api/backend proxy defined in
// next.config.ts. The browser never contacts the backend directly, so the backend
// URL is never baked into the client bundle and there are no CORS concerns — a
// down backend surfaces as a normal HTTP error instead of an opaque "Failed to
// fetch". Set BACKEND_URL in the Vercel project's Environment Variables to the
// Modal backend's public URL (see Docs/DEPLOYMENT.md). In local dev the proxy
// forwards to http://localhost:8000 (next.config.ts default).
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
  status: "uploading" | "uploaded" | "queued" | "running" | "succeeded" | "failed" | "cancelled";
  task?: Task;
  stage?: string;
  target?: string | null;
  error?: string;
}

export type Task = "auto" | "classification" | "regression" | "clustering" | "forecasting" | "image_classification";

export interface StartRunBody {
  task: Task;
  target?: string;
  date_column?: string;
  series_id_column?: string;
  horizon?: number;
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
  task?: Task;
  target?: string | null;
  problem_type?: "regression" | "classification" | "clustering" | "forecasting";
  accuracy_score?: number;
  score_metric?: string;
  higher_is_better?: boolean;
  plot_path?: string;
  justification?: string;
  model_name?: string;
  extra?: ResultExtra;
  error?: string;
  downloads?: DownloadKind[];
}

export type DownloadKind = "cleaned_csv" | "model" | "forecast";

// A plain link: the backend streams the file, or redirects to a short-lived
// R2 URL when R2 storage is configured.
export function downloadUrl(runId: string, kind: DownloadKind): string {
  return `${BASE}/runs/${runId}/download/${kind}`;
}

// Every upload PUTs straight to R2 from the browser, whatever its size — the
// backend and the frontend proxy never see the raw bytes. The one exception:
// if this backend has no R2 configured, /uploads/direct answers 409 and we
// fall back to the old proxy upload (≤30 MB) so local dev without R2 still works.
class R2NotConfiguredError extends Error {}

export async function uploadCSV(file: File): Promise<UploadResponse> {
  try {
    return await uploadDirect(file);
  } catch (e) {
    if (e instanceof R2NotConfiguredError) return uploadViaProxy(file);
    throw e;
  }
}

async function uploadViaProxy(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/upload`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function uploadDirect(file: File): Promise<UploadResponse> {
  const res = await fetch(`${BASE}/uploads/direct`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ filename: file.name, size: file.size }),
  });
  if (res.status === 409) throw new R2NotConfiguredError(await res.text());
  if (!res.ok) throw new Error(await res.text());
  const { run_id, upload_url } = await res.json();
  const put = await fetch(upload_url, { method: "PUT", body: file });
  if (!put.ok) throw new Error(`Upload to storage failed (${put.status}).`);
  const done = await fetch(`${BASE}/uploads/${run_id}/complete`, { method: "POST" });
  if (!done.ok) throw new Error(await done.text());
  return done.json();
}

export async function startRun(runId: string, body: StartRunBody): Promise<RunStatus> {
  const res = await fetch(`${BASE}/runs/${runId}/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function cancelRun(runId: string): Promise<RunStatus> {
  const res = await fetch(`${BASE}/runs/${runId}/cancel`, { method: "POST" });
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
  task?: Task;
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
  source_format?: string;
  columns: string[];
  n_columns: number;
  preview: Record<string, unknown>[];
}

export async function getPreview(runId: string): Promise<PreviewData> {
  const res = await fetch(`${BASE}/runs/${runId}/preview`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Recorded run state only: training runs on separate GPU containers, so no
// live utilisation numbers are reported (the old ones were not real).
export interface Diagnostics {
  status?: string;
  stage?: string | null;
  task?: Task | null;
  gpu?: string;
  gpu_calls: number;
  elapsed_seconds?: number | null;
}

export async function getDiagnostics(runId: string): Promise<Diagnostics> {
  const res = await fetch(`${BASE}/runs/${runId}/diagnostics`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}


