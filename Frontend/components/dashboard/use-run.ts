"use client";

import { useCallback, useEffect, useState } from "react";
import {
  ApiError,
  getDeployment,
  getDiagnostics,
  getModelSchema,
  getPreview,
  getResult,
  getStatus,
  type Deployment,
  type Diagnostics,
  type ModelSchema,
  type PreviewData,
  type RunResult,
  type RunStatus,
} from "@/lib/api";
import type { StageKey } from "./dashboard";

const errorText = (e: unknown, fallback: string) => (e instanceof Error && e.message ? e.message : fallback);

export const isRunActive = (s?: RunStatus | null) => s?.status === "queued" || s?.status === "running";
/** The backend could not read the uploaded file, so training never began. */
export const isUploadFailed = (s?: RunStatus | null) => s?.status === "failed" && !s.task && !s.stage;
/** Training has begun at some point, whatever became of it. */
export const hasStarted = (s?: RunStatus | null) =>
  !!s && s.status !== "uploading" && s.status !== "uploaded" && !isUploadFailed(s);
/** Training stopped without a result. */
export const isRunFailed = (s?: RunStatus | null) =>
  (s?.status === "failed" || s?.status === "cancelled") && !isUploadFailed(s);
/** Where a run opens without a stage in its link: wherever it has got to. */
export const landingStage = (s: RunStatus): StageKey =>
  s.status === "succeeded" ? "results" : hasStarted(s) ? "training" : "data";

// The result file is written a moment after the status flips to succeeded, so
// it is retried for about a minute before the stage reports it missing.
const RESULT_RETRY_MS = 2000;
const RESULT_RETRIES = 30;

/**
 * Everything the dashboard knows about one run, kept fresh: the status while
 * the backend reads the upload, then the preview, status and diagnostics while
 * it trains, then the result, model schema and deployment once it has
 * succeeded. Every stage reads from this one place, so moving between stages
 * never refetches.
 */
export function useRun(runId: string | null) {
  const [loadKey, setLoadKey] = useState(0);
  const [resultKey, setResultKey] = useState(0);

  const [preview, setPreview] = useState<PreviewData | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [status, setStatus] = useState<RunStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [notFound, setNotFound] = useState(false);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [resultError, setResultError] = useState<string | null>(null);
  const [schema, setSchema] = useState<ModelSchema | null>(null);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [deployment, setDeployment] = useState<Deployment | null>(null);

  useEffect(() => {
    setPreview(null);
    setPreviewError(null);
    setStatus(null);
    setStatusError(null);
    setNotFound(false);
    setDiagnostics(null);
    setResult(null);
    setResultError(null);
    setSchema(null);
    setSchemaError(null);
    setDeployment(null);
    if (!runId) return;

    let cancelled = false;
    getStatus(runId)
      .then((s) => !cancelled && setStatus(s))
      .catch((e) => {
        if (cancelled) return;
        setStatusError(errorText(e, "Failed to load the run status."));
        if (e instanceof ApiError && e.status === 404) setNotFound(true);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, loadKey]);

  const phase = status?.status;
  // The dataset exists once the backend has read the upload, unless it failed to.
  const datasetReady = !!status && phase !== "uploading" && !isUploadFailed(status);
  const finished = phase === "succeeded" || isRunFailed(status);

  useEffect(() => {
    if (!runId || !datasetReady) return;
    let cancelled = false;
    setPreviewError(null);
    getPreview(runId)
      .then((d) => !cancelled && setPreview(d))
      .catch((e) => {
        if (cancelled) return;
        setPreviewError(errorText(e, "Failed to load the dataset preview."));
        if (e instanceof ApiError && e.status === 404) setNotFound(true);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, datasetReady, loadKey]);

  // While the backend reads an upload: the status, until the file is ready.
  // While queued or running: the status for the stage, diagnostics for the facts.
  useEffect(() => {
    const training = phase === "queued" || phase === "running";
    if (!runId || (!training && phase !== "uploading")) return;
    let cancelled = false;
    const tick = async () => {
      const [s, d] = await Promise.allSettled([getStatus(runId), training ? getDiagnostics(runId) : Promise.resolve(null)]);
      if (cancelled) return;
      if (s.status === "fulfilled") setStatus(s.value);
      if (d.status === "fulfilled" && d.value) setDiagnostics(d.value);
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [runId, phase]);

  // Finished either way: one last diagnostics read for the hardware facts.
  useEffect(() => {
    if (!runId || !finished) return;
    let cancelled = false;
    getDiagnostics(runId)
      .then((d) => !cancelled && setDiagnostics(d))
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [runId, finished]);

  // Succeeded: load what the results and deploy stages show.
  useEffect(() => {
    if (!runId || phase !== "succeeded") return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let attempts = 0;
    setResultError(null);
    setSchemaError(null);
    const loadResult = async () => {
      try {
        const r = await getResult(runId);
        if (!cancelled) setResult(r);
      } catch (e) {
        if (cancelled) return;
        if (++attempts < RESULT_RETRIES) timer = setTimeout(loadResult, RESULT_RETRY_MS);
        else setResultError(errorText(e, "The run finished but its results could not be loaded."));
      }
    };
    loadResult();
    getModelSchema(runId)
      .then((s) => !cancelled && setSchema(s))
      .catch((e) => !cancelled && setSchemaError(errorText(e, "No trained model found for this run.")));
    getDeployment(runId)
      .then((d) => !cancelled && setDeployment(d))
      .catch(() => {});
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [runId, phase, resultKey]);

  const deployPhase = deployment?.status;
  useEffect(() => {
    if (!runId || deployPhase !== "deploying") return;
    const id = setInterval(async () => {
      try {
        setDeployment(await getDeployment(runId));
      } catch {
        /* transient; keep polling */
      }
    }, 2500);
    return () => clearInterval(id);
  }, [runId, deployPhase]);

  const reload = useCallback(() => setLoadKey((k) => k + 1), []);
  const retryResults = useCallback(() => setResultKey((k) => k + 1), []);

  return {
    preview,
    previewError,
    status,
    setStatus,
    statusError,
    notFound,
    reload,
    diagnostics,
    result,
    resultError,
    schema,
    schemaError,
    retryResults,
    deployment,
    setDeployment,
  };
}

export type RunState = ReturnType<typeof useRun>;
