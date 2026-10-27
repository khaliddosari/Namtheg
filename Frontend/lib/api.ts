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

/**
 * A failed request, with a message fit to show and the HTTP status (0 when the
 * server was unreachable). `answered` is true when the backend itself sent the
 * error, as opposed to a proxy in front of it failing.
 */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly answered = false
  ) {
    super(message);
    this.name = "ApiError";
  }
}

// FastAPI answers {"detail": "..."}, or {"detail": [{msg, loc}, ...]} for a
// validation error; a proxy in front of it answers with plain text or an HTML
// page. All of it becomes one readable sentence.
function apiError(status: number, text: string): ApiError {
  let message = "";
  let answered = false;
  try {
    const detail = JSON.parse(text)?.detail;
    if (typeof detail === "string") message = detail;
    else if (Array.isArray(detail)) message = detail.map((d) => d?.msg ?? String(d)).join("; ");
    answered = !!message;
  } catch {
    message = /<[a-z!?/]/i.test(text) ? "" : text.trim();
  }
  if (!message) message = status >= 500 ? "The server ran into a problem." : "The request was rejected.";
  if (message.length > 300) message = `${message.slice(0, 297)}...`;
  return new ApiError(`${message} (${status})`, status, answered);
}

async function send(input: string, init?: RequestInit): Promise<Response> {
  let res: Response;
  try {
    res = await fetch(input, init);
  } catch {
    throw new ApiError("Can't reach the server. Check your connection and try again.", 0);
  }
  if (!res.ok) throw apiError(res.status, await res.text().catch(() => ""));
  return res;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  return (await send(`${BASE}${path}`, init)).json();
}

const postJson = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});


export interface UploadResponse {
  run_id: string;
  filename: string;
  columns: string[];
  preview: Record<string, unknown>[];
}

export interface RunStatus {
  run_id: string;
  status: "uploading" | "uploaded" | "queued" | "running" | "succeeded" | "failed" | "cancelled";
  filename?: string;
  size?: number; // bytes, for direct uploads
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

/** How much of a file has been sent, for a progress bar. */
export interface UploadProgress {
  loaded: number;
  total: number;
}

// XMLHttpRequest rather than fetch: only it reports upload progress.
function sendWithProgress(
  method: string,
  url: string,
  body: XMLHttpRequestBodyInit,
  onProgress?: (p: UploadProgress) => void
): Promise<string> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(method, url);
    if (onProgress) {
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) onProgress({ loaded: e.loaded, total: e.total });
      };
    }
    xhr.onload = () =>
      xhr.status >= 200 && xhr.status < 300 ? resolve(xhr.responseText) : reject(apiError(xhr.status, xhr.responseText));
    xhr.onerror = () => reject(new ApiError("The upload was interrupted. Check your connection and try again.", 0));
    xhr.send(body);
  });
}

/**
 * Sends a dataset and resolves with the new run's id as soon as the file is
 * stored. `processing` settles when the backend has taken the file to read, or
 * turned it down (see completeUpload); the run's status says "uploading" until
 * the file is read.
 *
 * The file goes straight to R2 whatever its size, so neither the backend nor
 * the frontend proxy carries the bytes. A backend with no R2 answers 409, and
 * the file then goes through the proxy to /upload instead (30 MB cap), which
 * keeps local development without R2 working.
 */
export async function uploadDataset(
  file: File,
  onProgress?: (p: UploadProgress) => void
): Promise<{ runId: string; processing: Promise<void> }> {
  let direct: { run_id: string; upload_url: string };
  try {
    direct = await request("/uploads/direct", postJson({ filename: file.name, size: file.size }));
  } catch (e) {
    if (!(e instanceof ApiError && e.status === 409)) throw e;
    const form = new FormData();
    form.append("file", file);
    const res: UploadResponse = JSON.parse(await sendWithProgress("POST", `${BASE}/upload`, form, onProgress));
    return { runId: res.run_id, processing: Promise.resolve() };
  }
  try {
    await sendWithProgress("PUT", direct.upload_url, file, onProgress);
  } catch (e) {
    // Storage answers in XML; its status says enough.
    throw e instanceof ApiError && e.status ? new ApiError(`Upload to storage failed (${e.status}).`, e.status) : e;
  }
  return { runId: direct.run_id, processing: completeUpload(direct.run_id) };
}

// Proxies that stop waiting do so later than this: Next's rewrite proxy, which
// serves /api/backend in development, gives up at 30 seconds by default.
const PROXY_PATIENCE_MS = 25_000;

/**
 * Asks the backend to read a file that is in storage. The backend answers at
 * once and reads in the background; the run's status then says when the file
 * is ready, or why it couldn't be read. A backend from before that change
 * reads inside this one request, which takes minutes for a large file, and a
 * proxy in front of it can give up first while the backend carries on. So
 * this rejects only when the backend itself turns the file down, or when the
 * request fails too soon to be a proxy giving up.
 */
export function completeUpload(runId: string): Promise<void> {
  const started = Date.now();
  return send(`${BASE}/uploads/${runId}/complete`, { method: "POST" }).then(
    () => undefined,
    (e: unknown) => {
      if (e instanceof ApiError && !e.answered && Date.now() - started >= PROXY_PATIENCE_MS) return;
      throw e;
    }
  );
}

export const startRun = (runId: string, body: StartRunBody): Promise<RunStatus> =>
  request(`/runs/${runId}/start`, postJson(body));

export const cancelRun = (runId: string): Promise<RunStatus> => request(`/runs/${runId}/cancel`, { method: "POST" });

export const getStatus = (runId: string): Promise<RunStatus> => request(`/runs/${runId}/status`);

export const getResult = (runId: string): Promise<RunResult> => request(`/runs/${runId}/result`);

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

export const startDeploy = (runId: string): Promise<Deployment> => request(`/runs/${runId}/deploy`, { method: "POST" });

export const getDeployment = (runId: string): Promise<Deployment> => request(`/runs/${runId}/deployment`);

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

export const getModelSchema = (runId: string): Promise<ModelSchema> => request(`/runs/${runId}/model_schema`);

export const predict = (
  runId: string,
  payload: { features: Record<string, number | string | null> } | { rows: (number | string | null)[][] },
): Promise<PredictionResponse> => request(`/runs/${runId}/predict`, postJson(payload));

export interface PreviewData {
  run_id: string;
  filename?: string;
  source_format?: string;
  columns: string[];
  n_columns: number;
  preview: Record<string, unknown>[];
}

export const getPreview = (runId: string): Promise<PreviewData> => request(`/runs/${runId}/preview`);

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

export const getDiagnostics = (runId: string): Promise<Diagnostics> => request(`/runs/${runId}/diagnostics`);
