"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ArrowRight,
  Download,
  FileSpreadsheet,
  HeartPulse,
  Play,
  RotateCcw,
  RotateCw,
  Search,
  Table2,
  Target,
  TriangleAlert,
  Upload,
  X,
} from "lucide-react";
import {
  completeUpload,
  startRun,
  uploadDataset,
  type PreviewData,
  type RunStatus,
  type StartRunBody,
  type Task,
  type UploadProgress,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button, buttonVariants } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { Field, FieldDescription, FieldLabel } from "@/components/ui/field";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import {
  Combobox,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxInput,
  ComboboxItem,
  ComboboxList,
} from "@/components/ui/combobox";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Fact, Meter, Panel, PanelEmpty, PanelSkeleton, StatusDot } from "@/components/panel";
import { Segmented } from "@/components/segmented";
import { hasStarted, isUploadFailed, type RunState } from "./use-run";
import type { StageKey } from "./dashboard";

// Must match UPLOAD_EXTENSIONS in Backend/app/main.py (.zip = images, one folder per class).
const ACCEPTED_EXTENSIONS = [".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".json", ".jsonl", ".zip"];
const MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024; // uploads go straight to storage (R2)

type Mode = "predict" | "cluster" | "forecast";
// Where the dataset is: nothing yet, loading a linked run, being read on the
// server after an upload, failed, or here.
type DataState = "none" | "loading" | "reading" | "error" | "ready";

// Past this, a file still being read has probably stopped being read.
const READ_PATIENCE_S = 20 * 60;

const MODES: { value: Mode; label: string; short: string }[] = [
  { value: "predict", label: "Predict a column", short: "Predict" },
  { value: "cluster", label: "Find groups", short: "Group" },
  { value: "forecast", label: "Forecast over time", short: "Forecast" },
];

const modeFromTask = (task?: Task): Mode =>
  task === "clustering" ? "cluster" : task === "forecasting" ? "forecast" : "predict";

const isBlank = (v: unknown) => v === null || v === undefined || String(v).trim() === "" || String(v).trim() === "-";

function formatBytes(bytes: number) {
  if (bytes >= 1024 * 1024 * 1024) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 ** 2).toFixed(2)} MB`;
  return `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

const formatSeconds = (s: number) => (s >= 60 ? `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s` : `${s}s`);

/** Seconds since `running` turned true, ticking once a second. */
function useElapsed(running: boolean) {
  const [since, setSince] = useState<number | null>(null);
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!running) {
      setSince(null);
      return;
    }
    setSince(Date.now());
    const id = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(id);
  }, [running]);
  return since === null ? 0 : Math.floor((Date.now() - since) / 1000);
}

export function DataStage({
  runId,
  run,
  onUploaded,
  onStarted,
  onNewRun,
  goToStage,
}: {
  runId: string | null;
  run: RunState;
  onUploaded: (runId: string) => void;
  onStarted: (status: RunStatus) => void;
  onNewRun: () => void;
  goToStage: (key: StageKey) => void;
}) {
  const { preview, previewError, status, statusError, notFound, reload } = run;
  const profile = useProfile(preview);
  const uploadFailed = isUploadFailed(status);
  const loadError = previewError ?? statusError;
  const data: DataState = !runId
    ? "none"
    : preview
      ? "ready"
      : loadError || uploadFailed
        ? "error"
        : status?.status === "uploading"
          ? "reading"
          : "loading";
  const [mode, setMode] = useState<Mode>("predict");
  const [target, setTarget] = useState("");

  // The last column is the usual home of the label.
  useEffect(() => {
    if (preview?.columns?.length) setTarget((t) => t || preview.columns[preview.columns.length - 1]);
  }, [preview]);

  const started = hasStarted(status);
  // Once started, show what the run was actually given.
  const shownMode = started ? modeFromTask(status?.task) : mode;
  const shownTarget = started ? (status?.target ?? "") : target;

  return (
    <div className="grid flex-1 gap-3 md:grid-cols-2 lock:h-full lock:grid-cols-12 lock:grid-rows-[auto_minmax(0,1fr)]">
      <UploadPanel
        runId={runId}
        status={status}
        preview={preview}
        loadError={loadError}
        uploadError={uploadFailed ? (status?.error ?? "The server couldn't read this file.") : null}
        notFound={notFound}
        onRetry={reload}
        onUploaded={onUploaded}
        onNewRun={onNewRun}
        className="lock:col-span-4 lock:row-span-2"
      />
      <ConfigurePanel
        runId={runId}
        preview={preview}
        data={data}
        started={started}
        mode={mode}
        shownMode={shownMode}
        setMode={setMode}
        target={target}
        shownTarget={shownTarget}
        setTarget={setTarget}
        onStarted={onStarted}
        goToStage={goToStage}
        className="lock:col-span-5"
      />
      <ProfilePanel preview={preview} profile={profile} data={data} className="md:col-span-2 lock:col-span-3" />
      <PreviewPanel preview={preview} profile={profile} data={data} target={shownMode === "cluster" ? null : shownTarget || null} className="md:col-span-2 lock:col-span-8" />
    </div>
  );
}

/* ─── Upload ─────────────────────────────────────────────────────────── */

function UploadPanel({
  runId,
  status,
  preview,
  loadError,
  uploadError,
  notFound,
  onRetry,
  onUploaded,
  onNewRun,
  className,
}: {
  runId: string | null;
  status: RunStatus | null;
  preview: PreviewData | null;
  loadError: string | null;
  uploadError: string | null;
  notFound: boolean;
  onRetry: () => void;
  onUploaded: (runId: string) => void;
  onNewRun: () => void;
  className?: string;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [sending, setSending] = useState(false);
  const [progress, setProgress] = useState<UploadProgress | null>(null);
  const [error, setError] = useState<string | null>(null);
  // The backend turning down a stored file; asking it to read again may help.
  const [readError, setReadError] = useState<string | null>(null);

  // A file dropped anywhere else must not navigate the tab to it.
  useEffect(() => {
    const prevent = (e: DragEvent) => e.preventDefault();
    window.addEventListener("dragover", prevent);
    window.addEventListener("drop", prevent);
    return () => {
      window.removeEventListener("dragover", prevent);
      window.removeEventListener("drop", prevent);
    };
  }, []);

  const choose = useCallback((f: File) => {
    const ext = f.name.slice(f.name.lastIndexOf(".")).toLowerCase();
    if (!ACCEPTED_EXTENSIONS.includes(ext)) {
      setError(`Unsupported file type. Accepted: ${ACCEPTED_EXTENSIONS.join(", ")}`);
      return;
    }
    if (f.size === 0) {
      setError("That file is empty.");
      return;
    }
    if (f.size > MAX_UPLOAD_BYTES) {
      setError("File is too large. The limit is 2 GB.");
      return;
    }
    setError(null);
    setFile(f);
  }, []);

  // The run's status says when the file is ready; this only surfaces a
  // backend that says no.
  const watch = useCallback((reading: Promise<void>) => {
    setReadError(null);
    reading.catch((e) => setReadError(e instanceof Error ? e.message : "The server couldn't read the file."));
  }, []);

  const upload = async () => {
    if (!file) return;
    setSending(true);
    setProgress(null);
    setError(null);
    try {
      const { runId: id, processing } = await uploadDataset(file, setProgress);
      onUploaded(id);
      watch(processing);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Upload failed.");
    } finally {
      setSending(false);
    }
  };

  const readAgain = () => {
    if (runId) watch(completeUpload(runId));
  };

  const browse = () => document.getElementById("dataset-input")?.click();
  const uploaded = !!runId;
  const reading = uploaded && status?.status === "uploading";
  const waited = useElapsed(reading);
  const sentPct = progress && progress.total ? Math.round((progress.loaded / progress.total) * 100) : 0;
  const failure = uploadError ?? loadError;

  return (
    <Panel
      title="Upload"
      caption="CSV, TSV, Excel, Parquet, JSON, or a .zip of images with one folder per class"
      className={className}
    >
      <div
        onDragOver={(e) => {
          e.preventDefault();
          if (!uploaded) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          const f = e.dataTransfer.files[0];
          if (f && !uploaded && !sending) choose(f);
        }}
        className={cn(
          "flex min-h-52 flex-1 flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-primary/25 bg-muted/50 p-4 text-center transition-colors lock:min-h-0",
          dragging && "border-primary bg-primary/10"
        )}
      >
        {uploaded && failure ? (
          <div role="alert" className="flex flex-col items-center gap-3">
            <span className="flex size-12 items-center justify-center rounded-xl bg-red-500/15 text-red-300 ring-1 ring-red-400/30">
              <TriangleAlert className="size-6" />
            </span>
            <div className="min-w-0 max-w-full">
              <p className="text-base font-bold">
                {notFound ? "Run not found" : uploadError ? "The file couldn't be read" : "Couldn't load this dataset"}
              </p>
              <p className="text-xs break-words text-muted-foreground">
                {notFound ? `There is no run ${runId} on this server. It may have been removed.` : failure}
              </p>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-2">
              {!notFound && !uploadError && (
                <Button variant="outline" size="sm" onClick={onRetry} className="h-9 lg:h-7">
                  <RotateCw data-icon="inline-start" />
                  Try again
                </Button>
              )}
              <Button variant={notFound || uploadError ? "default" : "ghost"} size="sm" onClick={onNewRun} className="h-9 lg:h-7">
                <RotateCcw data-icon="inline-start" />
                Start a new run
              </Button>
            </div>
          </div>
        ) : reading ? (
          <div className="flex w-full max-w-sm flex-col items-center gap-3">
            <span className="flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary ring-1 ring-primary/25">
              <FileSpreadsheet className="size-6" />
            </span>
            <div className="min-w-0 max-w-full">
              <p className="truncate text-base font-bold" title={status?.filename}>
                {status?.filename ?? "Your file"}
              </p>
              <p className="flex items-center justify-center gap-2 text-sm text-primary">
                <StatusDot tone={readError ? "malfunction" : "running"} />
                {readError ? "The server stopped reading it" : "Reading it on the server"}
                <span className="font-mono text-xs text-muted-foreground tabular-nums">{formatSeconds(waited)}</span>
              </p>
            </div>
            {readError ? (
              <>
                <p role="alert" className="text-xs break-words text-red-300">
                  {readError}
                </p>
                <div className="flex flex-wrap items-center justify-center gap-2">
                  <Button variant="outline" size="sm" onClick={readAgain} className="h-9 lg:h-7">
                    <RotateCw data-icon="inline-start" />
                    Try again
                  </Button>
                  <Button variant="ghost" size="sm" onClick={onNewRun} className="h-9 lg:h-7">
                    <RotateCcw data-icon="inline-start" />
                    Start over
                  </Button>
                </div>
              </>
            ) : waited > READ_PATIENCE_S ? (
              <>
                <p className="text-xs text-muted-foreground">This is taking much longer than usual, so the server may have stopped.</p>
                <div className="flex flex-wrap items-center justify-center gap-2">
                  <Button variant="outline" size="sm" onClick={readAgain} className="h-9 lg:h-7">
                    <RotateCw data-icon="inline-start" />
                    Read it again
                  </Button>
                  <Button variant="ghost" size="sm" onClick={onNewRun} className="h-9 lg:h-7">
                    <RotateCcw data-icon="inline-start" />
                    Start over
                  </Button>
                </div>
              </>
            ) : (
              <p className="text-xs text-muted-foreground">
                {status?.size ? `${formatBytes(status.size)} is in storage. ` : "The file is in storage. "}
                Large files take a few minutes. You can leave this page; its link brings you back here.
              </p>
            )}
          </div>
        ) : uploaded ? (
          <>
            <span className="flex size-12 items-center justify-center rounded-xl bg-green-500/15 text-green-300 ring-1 ring-green-400/30">
              <FileSpreadsheet className="size-6" />
            </span>
            <div className="min-w-0 max-w-full">
              <p className="truncate text-base font-bold" title={preview?.filename}>
                {preview?.filename ?? "Dataset"}
              </p>
              <p className="font-mono text-xs text-muted-foreground" dir="ltr">
                run {runId?.slice(0, 12)}
              </p>
            </div>
            {preview ? (
              <dl className="grid w-full max-w-xs grid-cols-3 gap-2 text-start">
                <Fact label="Format" value={preview.source_format ?? "table"} />
                <Fact label="Columns" value={preview.n_columns} valueClassName="font-mono tabular-nums" />
                <Fact label="Preview rows" value={preview.preview.length} valueClassName="font-mono tabular-nums" />
              </dl>
            ) : (
              <PanelSkeleton rows={2} className="w-full max-w-xs flex-none items-center" />
            )}
            <Button variant="outline" size="sm" onClick={onNewRun} className="h-9 lg:h-7">
              <RotateCcw data-icon="inline-start" />
              Start over with another file
            </Button>
          </>
        ) : sending && file ? (
          <div className="flex w-full max-w-sm flex-col items-center gap-3">
            <span className="flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary ring-1 ring-primary/25">
              <Upload className="size-6" />
            </span>
            <div className="w-full min-w-0">
              <p className="truncate text-base font-bold" title={file.name}>
                Uploading {file.name}
              </p>
              <p className="text-xs text-muted-foreground">
                {progress && progress.loaded >= progress.total ? "Sent. Handing it to the server..." : "Straight to storage, then read on the server."}
              </p>
            </div>
            <div className="flex w-full flex-col gap-1.5">
              <div role="progressbar" aria-label="Upload progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={sentPct}>
                <Meter value={sentPct} />
              </div>
              <p className="flex justify-between font-mono text-xs text-muted-foreground tabular-nums">
                <span>{progress ? `${formatBytes(progress.loaded)} of ${formatBytes(progress.total)}` : "Starting..."}</span>
                <span>{sentPct}%</span>
              </p>
            </div>
          </div>
        ) : file ? (
          <>
            <span className="flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary ring-1 ring-primary/25">
              <FileSpreadsheet className="size-6" />
            </span>
            <div className="min-w-0 max-w-full">
              <p className="truncate text-base font-bold" title={file.name}>
                {file.name}
              </p>
              <p className="font-mono text-xs text-muted-foreground tabular-nums">{formatBytes(file.size)}</p>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-2">
              <Button variant="ghost" size="sm" onClick={() => setFile(null)} className="h-9 lg:h-7">
                <X data-icon="inline-start" />
                Remove
              </Button>
              <Button onClick={upload} className="h-10 lg:h-8">
                <Upload data-icon="inline-start" />
                Upload and profile
              </Button>
            </div>
          </>
        ) : (
          <>
            <span className="flex size-12 items-center justify-center rounded-xl bg-primary/10 text-primary ring-1 ring-primary/25 lock-short:hidden">
              <Upload className="size-6" />
            </span>
            <div>
              <p className="text-base font-bold">Drop a dataset here</p>
              <p className="text-xs text-muted-foreground">Checked on upload, then profiled on the server.</p>
            </div>
            <Button variant="outline" onClick={browse} className="h-10 lg:h-8">
              Browse files
            </Button>
          </>
        )}
        <input
          id="dataset-input"
          type="file"
          accept={ACCEPTED_EXTENSIONS.join(",")}
          className="sr-only"
          tabIndex={-1}
          aria-label="Choose a dataset file"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) choose(f);
            e.target.value = "";
          }}
        />
      </div>

      {error && (
        <p role="alert" className="text-xs text-red-300">
          {error}
        </p>
      )}

      <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-muted-foreground">
        No dataset handy?
        <a href="/insurance.csv" download="insurance.csv" className={cn(buttonVariants({ variant: "link", size: "sm" }), "h-auto px-0")}>
          <Download data-icon="inline-start" />
          Download a sample (insurance.csv)
        </a>
      </p>
    </Panel>
  );
}

/* ─── Configure run ──────────────────────────────────────────────────── */

function ConfigurePanel({
  runId,
  preview,
  data,
  started,
  mode,
  shownMode,
  setMode,
  target,
  shownTarget,
  setTarget,
  onStarted,
  goToStage,
  className,
}: {
  runId: string | null;
  preview: PreviewData | null;
  data: DataState;
  started: boolean;
  mode: Mode;
  shownMode: Mode;
  setMode: (mode: Mode) => void;
  target: string;
  shownTarget: string;
  setTarget: (target: string) => void;
  onStarted: (status: RunStatus) => void;
  goToStage: (key: StageKey) => void;
  className?: string;
}) {
  const columns = preview?.columns ?? [];
  const [dateColumn, setDateColumn] = useState("");
  const [seriesColumn, setSeriesColumn] = useState("");
  const [horizon, setHorizon] = useState("");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // A date or series column can't also be the value being forecast.
  useEffect(() => {
    if (dateColumn && dateColumn === target) setDateColumn("");
    if (seriesColumn && (seriesColumn === target || seriesColumn === dateColumn)) setSeriesColumn("");
  }, [target, dateColumn, seriesColumn]);

  const isImages = preview?.source_format === "images";
  const horizonSteps = horizon.trim() === "" ? undefined : Number(horizon);
  const horizonValid = horizonSteps === undefined || (Number.isInteger(horizonSteps) && horizonSteps >= 1 && horizonSteps <= 1000);
  // Why Start is disabled, said next to it rather than left to guesswork.
  const blocker =
    isImages || mode === "cluster"
      ? null
      : !target
        ? "Pick a target column to start."
        : mode === "forecast" && !dateColumn
          ? "Pick the date column to start."
          : mode === "forecast" && !horizonValid
            ? "The horizon must be a whole number from 1 to 1000."
            : null;
  const canStart = !blocker;

  const body = (): StartRunBody => {
    if (isImages) return { task: "image_classification" };
    if (mode === "cluster") return { task: "clustering" };
    if (mode === "forecast") {
      return {
        task: "forecasting",
        target,
        date_column: dateColumn,
        series_id_column: seriesColumn || undefined,
        horizon: horizonSteps,
      };
    }
    return { task: "auto", target };
  };

  const start = async () => {
    if (!runId || !canStart) return;
    setStarting(true);
    setError(null);
    try {
      onStarted(await startRun(runId, body()));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start the run.");
    } finally {
      setStarting(false);
    }
  };

  return (
    <Panel
      title="Configure run"
      caption="Pick what to learn. Problem type, feature handling and missing values are detected automatically."
      className={className}
    >
      {data === "loading" ? (
        <PanelSkeleton rows={4} />
      ) : !preview ? (
        <PanelEmpty
          icon={<Target />}
          title={data === "error" ? "No dataset to configure" : data === "reading" ? "Almost ready" : "Waiting for a dataset"}
          description={
            data === "error"
              ? "The dataset for this run didn't load."
              : data === "reading"
                ? "Choose a target as soon as the server has read your file."
                : "Upload a file to choose a target and start training."
          }
        />
      ) : (
        <>
          {isImages ? (
            <p className="text-sm text-muted-foreground">
              Image classification: each folder in the zip is a class, and pretrained CNNs are fine-tuned on the GPU to tell
              them apart.
            </p>
          ) : (
            <>
              <Segmented
                label="What to learn"
                value={shownMode}
                onChange={setMode}
                options={MODES}
                disabled={started}
                className="w-full sm:w-fit"
                itemClassName="flex-1 sm:flex-none"
              />

              {shownMode === "cluster" ? (
                <p className="text-sm text-muted-foreground">
                  No target: every column is used to find natural groups (K-Means, HDBSCAN, DBSCAN, Spectral), and each row
                  gets a cluster.
                </p>
              ) : started ? (
                <dl>
                  <Fact
                    label={shownMode === "forecast" ? "Value forecast" : "Target column"}
                    value={shownTarget || "-"}
                    title={shownTarget}
                    valueClassName="font-mono text-base"
                  />
                </dl>
              ) : (
                <Field>
                  <FieldLabel htmlFor="target-column">{shownMode === "forecast" ? "Value to forecast" : "Target column"}</FieldLabel>
                  <Combobox
                    items={columns}
                    value={target || null}
                    onValueChange={(v) => v && setTarget(v as string)}
                  >
                    <ComboboxInput id="target-column" placeholder="Search columns" className="h-10 w-full font-mono lg:h-8" />
                    <ComboboxContent>
                      <ComboboxEmpty>No column matches.</ComboboxEmpty>
                      <ComboboxList>
                        {(col: string) => (
                          <ComboboxItem key={col} value={col} className="font-mono">
                            {col}
                          </ComboboxItem>
                        )}
                      </ComboboxList>
                    </ComboboxContent>
                  </Combobox>
                  <FieldDescription className="text-xs lock-short:hidden">
                    {shownMode === "forecast" ? "The numeric series the model projects forward." : "The column the model learns to predict."}
                  </FieldDescription>
                </Field>
              )}

              {shownMode === "forecast" && !started && (
                <div className="grid gap-3 sm:grid-cols-3">
                  <Field>
                    <FieldLabel htmlFor="date-column">Date column</FieldLabel>
                    <NativeSelect id="date-column" value={dateColumn} onChange={(e) => setDateColumn(e.target.value)} className="h-10 lg:h-8">
                      <NativeSelectOption value="">Choose...</NativeSelectOption>
                      {columns
                        .filter((c) => c !== target)
                        .map((c) => (
                          <NativeSelectOption key={c} value={c}>
                            {c}
                          </NativeSelectOption>
                        ))}
                    </NativeSelect>
                  </Field>
                  <Field>
                    <FieldLabel htmlFor="series-column">Series id</FieldLabel>
                    <NativeSelect id="series-column" value={seriesColumn} onChange={(e) => setSeriesColumn(e.target.value)} className="h-10 lg:h-8">
                      <NativeSelectOption value="">One series</NativeSelectOption>
                      {columns
                        .filter((c) => c !== target && c !== dateColumn)
                        .map((c) => (
                          <NativeSelectOption key={c} value={c}>
                            {c}
                          </NativeSelectOption>
                        ))}
                    </NativeSelect>
                  </Field>
                  <Field>
                    <FieldLabel htmlFor="horizon">Horizon (steps)</FieldLabel>
                    <Input
                      id="horizon"
                      type="number"
                      min={1}
                      max={1000}
                      inputMode="numeric"
                      dir="ltr"
                      value={horizon}
                      onChange={(e) => setHorizon(e.target.value)}
                      placeholder="30"
                      aria-invalid={!horizonValid || undefined}
                      aria-describedby="horizon-help"
                      className="h-10 font-mono lg:h-8"
                    />
                    <FieldDescription id="horizon-help" className="text-xs">
                      Optional. Leave empty to pick from the data&apos;s frequency.
                    </FieldDescription>
                  </Field>
                </div>
              )}
            </>
          )}

          <div className="mt-auto flex flex-wrap items-center justify-end gap-3 pt-1">
            {error ? (
              <p role="alert" className="me-auto text-xs text-red-300">
                {error}
              </p>
            ) : (
              !started &&
              blocker && (
                <p id="start-blocker" className="me-auto text-xs text-muted-foreground">
                  {blocker}
                </p>
              )
            )}
            {started ? (
              <Button variant="outline" onClick={() => goToStage("training")} className="h-10 lg:h-8">
                View training
                <ArrowRight data-icon="inline-end" />
              </Button>
            ) : (
              <Button
                onClick={start}
                disabled={!canStart || starting}
                aria-describedby={blocker ? "start-blocker" : undefined}
                className="h-10 lg:h-8"
              >
                {starting ? <Spinner data-icon="inline-start" /> : <Play data-icon="inline-start" />}
                {starting ? "Starting..." : "Start training"}
              </Button>
            )}
          </div>
        </>
      )}
    </Panel>
  );
}

/* ─── Profile maths (from the preview rows only) ────────────────────────── */

interface ColumnStats {
  column: string;
  numeric: boolean;
  mean: string;
  std: string;
  min: string;
  max: string;
  nullPct: string;
  unique: number;
}

function useProfile(preview: PreviewData | null) {
  return useMemo(() => {
    if (!preview) return null;
    const rows = preview.preview;
    const numericCols = preview.columns.filter((col) => {
      const v = rows[0]?.[col];
      return typeof v === "number" || (typeof v === "string" && v.trim() !== "" && !isNaN(Number(v)));
    });
    const numericSet = new Set(numericCols);

    const stats: ColumnStats[] = preview.columns.map((col) => {
      const values = rows.map((r) => r[col]);
      const blanks = values.filter(isBlank).length;
      const nullPct = values.length ? `${((blanks / values.length) * 100).toFixed(1)}%` : "0%";
      const unique = new Set(values.filter((v) => !isBlank(v)).map(String)).size;
      const nums = values.filter((v) => !isBlank(v)).map(Number).filter((n) => !isNaN(n));
      if (numericSet.has(col) && nums.length) {
        const mean = nums.reduce((a, b) => a + b, 0) / nums.length;
        const std = Math.sqrt(nums.reduce((a, b) => a + (b - mean) ** 2, 0) / nums.length);
        return {
          column: col,
          numeric: true,
          mean: mean.toFixed(2),
          std: std.toFixed(2),
          min: Math.min(...nums).toFixed(2),
          max: Math.max(...nums).toFixed(2),
          nullPct,
          unique,
        };
      }
      return { column: col, numeric: false, mean: "-", std: "-", min: "-", max: "-", nullPct, unique };
    });

    const totalCells = rows.length * preview.columns.length;
    const nullCells = preview.columns.reduce((acc, col) => acc + rows.filter((r) => isBlank(r[col])).length, 0);
    const nullRows = rows.filter((r) => preview.columns.some((col) => isBlank(r[col]))).length;
    const health = totalCells > 0 ? Math.round(((totalCells - nullCells) / totalCells) * 100) : 100;

    return { stats, numericSet, numeric: numericCols.length, categorical: preview.columns.length - numericCols.length, nullRows, nullCells, health };
  }, [preview]);
}

type Profile = ReturnType<typeof useProfile>;

/* ─── Data profile ───────────────────────────────────────────────────── */

function ProfilePanel({
  preview,
  profile,
  data,
  className,
}: {
  preview: PreviewData | null;
  profile: Profile;
  data: DataState;
  className?: string;
}) {
  const numericPct = profile && preview?.n_columns ? (profile.numeric / preview.n_columns) * 100 : 0;

  return (
    <Panel
      title="Data profile"
      caption={preview ? `From the first ${preview.preview.length.toLocaleString()} rows` : "Completeness and column types"}
      className={className}
    >
      {data === "loading" ? (
        <PanelSkeleton rows={4} />
      ) : !preview || !profile ? (
        <PanelEmpty
          icon={<HeartPulse />}
          title="No profile yet"
          description={data === "reading" ? "Completeness and column types appear once the file is read." : "Completeness and column types appear after upload."}
        />
      ) : (
        <div className="@container min-h-0 flex-1 overflow-y-auto">
          <div className="grid gap-4 @lg:grid-cols-2">
            <section className="flex flex-col gap-2">
              <div className="flex items-baseline justify-between gap-2">
                <h3 className="text-xs font-medium text-muted-foreground">Completeness</h3>
                <span className="font-mono text-2xl leading-none font-bold tabular-nums">{profile.health}%</span>
              </div>
              <Meter value={profile.health} />
              <dl className="grid grid-cols-2 gap-2 lock-short:hidden">
                <Fact label="Rows with gaps" value={profile.nullRows} valueClassName="font-mono tabular-nums" />
                <Fact label="Empty cells" value={profile.nullCells} valueClassName="font-mono tabular-nums" />
              </dl>
            </section>
            <section className="flex flex-col gap-2">
              <div className="flex items-baseline justify-between gap-2">
                <h3 className="text-xs font-medium text-muted-foreground">Columns</h3>
                <span className="font-mono text-2xl leading-none font-bold tabular-nums">{preview.n_columns}</span>
              </div>
              <div className="flex h-1.5 w-full overflow-hidden rounded-full bg-muted">
                <div className="h-full bg-chart-1" style={{ width: `${numericPct}%` }} />
                <div className="h-full bg-chart-3" style={{ width: `${100 - numericPct}%` }} />
              </div>
              <dl className="grid grid-cols-2 gap-2 lock-short:hidden">
                <Fact
                  label="Numeric"
                  value={
                    <span className="inline-flex items-center gap-1.5">
                      <span aria-hidden="true" className="size-2 rounded-full bg-chart-1" />
                      <span className="font-mono tabular-nums">{profile.numeric}</span>
                    </span>
                  }
                />
                <Fact
                  label="Categorical"
                  value={
                    <span className="inline-flex items-center gap-1.5">
                      <span aria-hidden="true" className="size-2 rounded-full bg-chart-3" />
                      <span className="font-mono tabular-nums">{profile.categorical}</span>
                    </span>
                  }
                />
              </dl>
            </section>
          </div>
        </div>
      )}
    </Panel>
  );
}

/* ─── Dataset preview ────────────────────────────────────────────────── */

function PreviewPanel({
  preview,
  profile,
  data,
  target,
  className,
}: {
  preview: PreviewData | null;
  profile: Profile;
  data: DataState;
  target: string | null;
  className?: string;
}) {
  const [view, setView] = useState<"rows" | "stats">("rows");
  const [query, setQuery] = useState("");

  const rows = useMemo(() => {
    if (!preview) return [];
    const q = query.trim().toLowerCase();
    if (!q) return preview.preview;
    return preview.preview.filter((row) => preview.columns.some((col) => String(row[col] ?? "").toLowerCase().includes(q)));
  }, [preview, query]);

  return (
    <Panel
      title={view === "rows" ? "Dataset preview" : "Column statistics"}
      caption={
        preview
          ? view === "rows"
            ? `${rows.length.toLocaleString()} of ${preview.preview.length.toLocaleString()} preview rows`
            : "Computed from the preview rows"
          : "Raw rows and per-column statistics"
      }
      className={className}
    >
      {data === "loading" ? (
        <PanelSkeleton rows={6} />
      ) : !preview || !profile ? (
        <PanelEmpty
          icon={<Table2 />}
          title="Nothing to preview"
          description={data === "reading" ? "The first rows show here once the file is read." : "The first rows of your dataset show here once it is uploaded."}
        />
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2">
            <Segmented
              label="Table view"
              value={view}
              onChange={setView}
              options={[
                { value: "rows", label: "Raw rows" },
                { value: "stats", label: "Summary stats" },
              ]}
            />
            {view === "rows" && (
              <div className="relative ms-auto w-full sm:w-56">
                <Search aria-hidden="true" className="pointer-events-none absolute start-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  aria-label="Filter rows"
                  placeholder="Filter rows"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="h-10 ps-8 lg:h-7"
                />
              </div>
            )}
          </div>

          <div
            tabIndex={0}
            role="region"
            aria-label={view === "rows" ? "Dataset rows" : "Column statistics"}
            className="max-h-[26rem] min-h-0 flex-1 overflow-auto rounded-xl bg-background/30 ring-1 ring-border outline-none focus-visible:ring-3 focus-visible:ring-ring/50 lock:max-h-none *:data-[slot=table-container]:overflow-visible"
          >
            {view === "rows" ? (
              <Table className="text-xs">
                <TableHeader className="sticky top-0 z-10 bg-popover backdrop-blur-xl">
                  <TableRow className="hover:bg-transparent">
                    {preview.columns.map((col) => (
                      <TableHead key={col} className={cn("h-9 px-3 text-xs text-muted-foreground", col === target && "text-primary")}>
                        <span className="font-mono" title={col === target ? "Target column" : undefined}>
                          {col}
                        </span>
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody className="font-mono">
                  {rows.length ? (
                    rows.map((row, i) => (
                      <TableRow key={i} className="border-border/60">
                        {preview.columns.map((col) => (
                          <TableCell
                            key={col}
                            className={cn(
                              "max-w-56 truncate px-3 py-1.5",
                              profile.numericSet.has(col) && "text-end tabular-nums",
                              col === target && "bg-primary/10 font-bold text-primary"
                            )}
                            title={String(row[col] ?? "")}
                          >
                            {String(row[col] ?? "-")}
                          </TableCell>
                        ))}
                      </TableRow>
                    ))
                  ) : (
                    <TableRow>
                      <TableCell colSpan={preview.columns.length} className="py-8 text-center font-sans text-muted-foreground">
                        No rows match the filter.
                      </TableCell>
                    </TableRow>
                  )}
                </TableBody>
              </Table>
            ) : (
              <Table className="text-xs">
                <TableHeader className="sticky top-0 z-10 bg-popover backdrop-blur-xl">
                  <TableRow className="hover:bg-transparent">
                    <TableHead className="h-9 px-3 text-xs text-muted-foreground">Column</TableHead>
                    <TableHead className="h-9 px-3 text-xs text-muted-foreground">Type</TableHead>
                    {["Mean", "Std dev", "Min", "Max", "Null %", "Unique"].map((h) => (
                      <TableHead key={h} className="h-9 px-3 text-end text-xs text-muted-foreground">
                        {h}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {profile.stats.map((s) => (
                    <TableRow key={s.column} className={cn("border-border/60", s.column === target && "bg-primary/10")}>
                      <TableCell className={cn("max-w-56 truncate px-3 py-1.5 font-mono font-medium", s.column === target && "text-primary")} title={s.column}>
                        {s.column}
                      </TableCell>
                      <TableCell className="px-3 py-1.5">
                        <span className="inline-flex items-center gap-1.5 text-muted-foreground">
                          <span aria-hidden="true" className={cn("size-2 rounded-full", s.numeric ? "bg-chart-1" : "bg-chart-3")} />
                          {s.numeric ? "numeric" : "categorical"}
                        </span>
                      </TableCell>
                      {[s.mean, s.std, s.min, s.max, s.nullPct, String(s.unique)].map((v, i) => (
                        <TableCell key={i} className="px-3 py-1.5 text-end font-mono tabular-nums">
                          {v}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </div>
        </>
      )}
    </Panel>
  );
}
