"use client";

import { useRef, useState } from "react";
import {
  ArrowRight,
  ChartColumn,
  Download,
  FileText,
  Image as ImageIcon,
  ListOrdered,
  Package,
  ScanSearch,
  SlidersHorizontal,
  Table2,
  Trophy,
  TriangleAlert,
  ZoomIn,
} from "lucide-react";
import { downloadUrl, plotUrl, type RunResult, type TuningTrial } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button, buttonVariants } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { toast } from "@/components/ui/toast";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Brand, BrandMark, formatArabicBrand } from "@/components/brand";
import { Fact, Meter, Panel, PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel";
import { Segmented } from "@/components/segmented";
import { isRunFailed, type RunState } from "./use-run";
import { exportReportPdf, exportReportPng } from "./report-export";
import { AnalysisDialog, hasAnalysis } from "./analysis-dialog";
import { formatScore, metricInfo } from "./metrics";
import type { StageKey } from "./dashboard";

const PLOT_TITLES: Record<string, { title: string; caption: string }> = {
  classification: { title: "Confusion matrix", caption: "Diagonal cells are correct predictions." },
  regression: { title: "Predicted vs actual", caption: "Points close to the diagonal are accurate predictions." },
};

export function ResultsStage({
  runId,
  run,
  goToStage,
}: {
  runId: string | null;
  run: RunState;
  goToStage: (key: StageKey) => void;
}) {
  const { status, statusError, result, resultError, retryResults } = run;
  const reportRef = useRef<HTMLDivElement>(null);
  const failed = isRunFailed(status);
  // Loading a linked run, or reading a finished one's results.
  const waiting = (!!runId && !status && !statusError) || (status?.status === "succeeded" && !result && !resultError);

  // Every card waits the same way before there is a result.
  const empty = (icon: React.ReactNode, title: string, rows = 4) =>
    waiting ? (
      <PanelSkeleton rows={rows} />
    ) : (
      <PanelEmpty
        icon={icon}
        title={resultError ? "Results didn't load" : title}
        description={
          resultError
            ? "Try again from the Champion model card."
            : failed
              ? "This run did not finish, so there is nothing to report."
              : "Appears when training finishes."
        }
      />
    );

  // The export lays the cards out as a document instead (see report-export.ts):
  // data-export-span is each card's width out of twelve there. The plot and
  // the feature importances share a row; the plot is square, so its share
  // follows the length of the list beside it and the two come out level.
  const featureCount = Math.min(10, result?.extra?.top_features?.length ?? 0);
  const plotSpan = featureCount >= 9 ? 6 : featureCount >= 6 ? 5 : 4;

  return (
    <div ref={reportRef} className="flex flex-1 flex-col @container/report lock:h-full">
      {result && runId && <ReportHeader runId={runId} result={result} filename={status?.filename} />}
      <div
        data-export-grid
        className="grid flex-1 gap-3 @3xl/report:grid-cols-2 @5xl/report:grid-cols-12 lock:h-full lock:grid-rows-[auto_minmax(0,1fr)]"
      >
        <ChampionPanel
          result={result}
          failed={failed}
          cancelled={status?.status === "cancelled"}
          error={status?.error}
          resultError={resultError}
          onRetry={retryResults}
          empty={empty(<Trophy />, "No champion yet", 3)}
          goToStage={goToStage}
        />
        <ScorePanel result={result} empty={empty(<ChartColumn />, "No score yet")} />
        <FilesPanel runId={runId} result={result} waiting={waiting} reportRef={reportRef} goToStage={goToStage} />
        <PlotPanel
          runId={runId}
          result={result}
          empty={waiting ? <PanelSkeleton rows={0} block /> : empty(<ScanSearch />, "No plot yet")}
          exportSpan={plotSpan}
        />
        <FeaturesPanel result={result} empty={empty(<SlidersHorizontal />, "No features yet", 6)} exportSpan={12 - plotSpan} />
        <ModelsPanel result={result} empty={empty(<ListOrdered />, "No models compared yet", 6)} />
      </div>
    </div>
  );
}

/* ─── Report header (only the export shows it) ───────────────────────── */

function ReportHeader({ runId, result, filename }: { runId: string; result: RunResult; filename?: string }) {
  const about = ["Model report", filename, result.target && `predicting ${result.target}`, result.problem_type].filter(Boolean).join(" · ");
  return (
    <header hidden data-export-show className="mb-3 flex items-center gap-4 rounded-2xl bg-card px-5 py-4 shadow-(--glass-shadow) ring-1 ring-(--glass-edge)">
      <span className="flex size-12 shrink-0 items-center justify-center rounded-xl bg-primary/10 ring-1 ring-primary/25">
        <BrandMark className="size-7" />
      </span>
      <div className="min-w-0 flex-1">
        <p className="flex items-baseline gap-2.5 text-2xl font-bold">
          <Brand />
          <span>Namtheg</span>
        </p>
        <p className="mt-1 text-sm text-muted-foreground">{about}</p>
      </div>
      <dl className="shrink-0 text-xs text-muted-foreground">
        <div className="flex justify-end gap-1.5">
          <dt>Run</dt>
          <dd className="font-mono text-foreground">{runId}</dd>
        </div>
        <div className="mt-1 flex justify-end gap-1.5">
          <dt>Exported</dt>
          <dd data-export-date className="text-foreground" />
        </div>
      </dl>
    </header>
  );
}

/* ─── Champion model ─────────────────────────────────────────────────── */

function ChampionPanel({
  result,
  failed,
  cancelled,
  error,
  resultError,
  onRetry,
  empty,
  goToStage,
}: {
  result: RunResult | null;
  failed: boolean;
  cancelled: boolean;
  error?: string | null;
  resultError: string | null;
  onRetry: () => void;
  empty: React.ReactNode;
  goToStage: (key: StageKey) => void;
}) {
  const [open, setOpen] = useState(false);
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const analysisShown = !!result && hasAnalysis(result);
  const modelName = result?.model_name ?? "Best model";
  const facts = [result?.problem_type, result?.extra?.hardware?.gpu].filter(Boolean).join(" · ");

  return (
    <Panel
      title="Champion model"
      caption={result?.target ? `Predicting ${result.target}` : "The model the pipeline picked, and why"}
      data-export-span="8"
      className="@3xl/report:col-span-2 @5xl/report:col-span-5"
    >
      {result ? (
        <>
          <div className="flex min-w-0 items-center gap-3">
            <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary ring-1 ring-primary/25 lock-short:hidden">
              <Trophy className="size-5" />
            </span>
            <div className="min-w-0">
              <p className="font-mono text-lg font-bold sm:text-xl lock:truncate" title={modelName}>
                {modelName}
              </p>
              {facts && <p className="truncate text-xs text-muted-foreground capitalize">{facts}</p>}
            </div>
          </div>
          {result.justification && (
            <>
              <p className="text-sm leading-relaxed text-muted-foreground lock:line-clamp-2 lock-short:line-clamp-1">
                {formatArabicBrand(result.justification)}
              </p>
              <Dialog open={open} onOpenChange={setOpen}>
                <DialogContent className="sm:max-w-xl">
                  <DialogHeader>
                    <DialogTitle>Why {modelName}</DialogTitle>
                    <DialogDescription render={<div />} className="max-h-[60vh] overflow-y-auto text-sm leading-relaxed">
                      {formatArabicBrand(result.justification)}
                    </DialogDescription>
                  </DialogHeader>
                </DialogContent>
              </Dialog>
            </>
          )}
          {(result.justification || analysisShown) && (
            // On locked pages the clamped rationale needs its own link; the analysis always does.
            <div className={cn("flex-wrap items-center gap-x-4 gap-y-1", analysisShown ? "flex" : "hidden lock:flex")} data-export-ignore>
              {result.justification && (
                <Button variant="link" size="sm" onClick={() => setOpen(true)} className="hidden h-auto px-0 lock:inline-flex">
                  Read the full rationale
                </Button>
              )}
              {analysisShown && (
                <Button variant="link" size="sm" onClick={() => setAnalysisOpen(true)} className="h-auto px-0">
                  What the analyst found
                </Button>
              )}
            </div>
          )}
          {analysisShown && <AnalysisDialog result={result} open={analysisOpen} onOpenChange={setAnalysisOpen} />}
        </>
      ) : resultError ? (
        <PanelError title="The results didn't load" message={resultError} onRetry={onRetry} />
      ) : failed ? (
        <PanelEmpty
          tone="error"
          icon={<TriangleAlert />}
          title={cancelled ? "The run was cancelled" : "The run failed"}
          description={error ?? "No model was selected."}
          action={
            <Button variant="outline" onClick={() => goToStage("training")} className="h-10 lg:h-8">
              See the pipeline
            </Button>
          }
        />
      ) : (
        empty
      )}
    </Panel>
  );
}

/* ─── Score ──────────────────────────────────────────────────────────── */

function ScorePanel({ result, empty }: { result: RunResult | null; empty: React.ReactNode }) {
  const metric = result?.score_metric ?? "score";
  const info = metricInfo(metric);
  const score = result?.accuracy_score ?? 0;
  const extra = result?.extra ?? {};
  const trainScore = extra.train_score ?? extra.train_accuracy ?? extra.train_r2;
  const testScore = extra.test_score;
  const overfitGap = extra.overfit_gap;
  const pct = score * 100;

  return (
    <Panel
      title={result ? info.short : "Score"}
      caption={result ? info.long : "The selection metric on held-out data"}
      data-export-span="4"
      className="@5xl/report:col-span-3"
    >
      {result ? (
        <>
          <p className="font-mono text-4xl leading-none font-bold tabular-nums lock-short:text-3xl" dir="ltr">
            {formatScore(score, metric)}
          </p>
          {info.percent && <Meter value={pct} barClassName="bg-chart-1" />}
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 lock-short:gap-y-1">
            {trainScore !== undefined && <Fact label="Train" value={formatScore(trainScore, metric, 1)} valueClassName="font-mono tabular-nums" />}
            {testScore !== undefined && <Fact label="Test" value={formatScore(testScore, metric, 1)} valueClassName="font-mono tabular-nums" />}
          </dl>
          {overfitGap !== undefined && overfitGap > 0.12 && (
            <p className="flex items-center gap-1.5 text-xs font-medium text-red-300">
              <TriangleAlert aria-hidden="true" className="size-3.5 shrink-0" />
              Overfit gap {(overfitGap * 100).toFixed(0)}%: train and test scores are far apart
            </p>
          )}
        </>
      ) : (
        empty
      )}
    </Panel>
  );
}

/* ─── Report and files (kept out of the exported image) ──────────────── */

function FilesPanel({
  runId,
  result,
  waiting,
  reportRef,
  goToStage,
}: {
  runId: string | null;
  result: RunResult | null;
  waiting: boolean;
  reportRef: React.RefObject<HTMLDivElement | null>;
  goToStage: (key: StageKey) => void;
}) {
  const [exporting, setExporting] = useState<"png" | "pdf" | null>(null);
  const downloads = result?.downloads ?? [];

  const exportAs = async (kind: "png" | "pdf") => {
    if (!reportRef.current || !runId || exporting) return;
    setExporting(kind);
    const name = `namtheg-${runId}-report.${kind}`;
    try {
      if (kind === "png") await exportReportPng(reportRef.current, name);
      else await exportReportPdf(reportRef.current, name, `Namtheg model report: ${result?.model_name ?? "champion"} for ${result?.target ?? runId}`);
      toast.add({ title: `Report saved as ${kind.toUpperCase()}`, description: name, type: "success" });
    } catch (e) {
      console.warn(`${kind.toUpperCase()} export failed`, e);
      toast.add({ title: "Export failed", description: "The report could not be rendered. Try again.", type: "error" });
    } finally {
      setExporting(null);
    }
  };

  const fileLink = cn(buttonVariants({ variant: "outline", size: "sm" }), "h-9 justify-start lg:h-7");

  return (
    <Panel
      title="Report and files"
      caption="Export this page, download the data and the model"
      data-export-ignore
      className="@5xl/report:col-span-4"
    >
      {waiting ? (
        <PanelSkeleton rows={3} />
      ) : !result || !runId ? (
        <PanelEmpty icon={<Package />} title="Nothing to download yet" description="The report, cleaned data and model package appear here." />
      ) : (
        <>
          <div className="grid grid-cols-2 gap-2">
            <Button variant="outline" size="sm" onClick={() => exportAs("png")} disabled={!!exporting} className="h-9 lg:h-7">
              {exporting === "png" ? <Spinner data-icon="inline-start" /> : <ImageIcon data-icon="inline-start" />}
              Export PNG
            </Button>
            <Button variant="outline" size="sm" onClick={() => exportAs("pdf")} disabled={!!exporting} className="h-9 lg:h-7">
              {exporting === "pdf" ? <Spinner data-icon="inline-start" /> : <FileText data-icon="inline-start" />}
              Export PDF
            </Button>
            {downloads.includes("cleaned_csv") && (
              <a href={downloadUrl(runId, "cleaned_csv")} className={fileLink}>
                <Table2 data-icon="inline-start" />
                Cleaned CSV
              </a>
            )}
            {downloads.includes("model") && (
              <a href={downloadUrl(runId, "model")} className={fileLink}>
                <Download data-icon="inline-start" />
                Model + weights
              </a>
            )}
          </div>
          <Button onClick={() => goToStage("deploy")} className="mt-auto h-10 lg:h-8">
            Deploy and try this model
            <ArrowRight data-icon="inline-end" />
          </Button>
        </>
      )}
    </Panel>
  );
}

/* ─── Plot ───────────────────────────────────────────────────────────── */

function PlotPanel({
  runId,
  result,
  empty,
  exportSpan,
}: {
  runId: string | null;
  result: RunResult | null;
  empty: React.ReactNode;
  exportSpan: number;
}) {
  const [open, setOpen] = useState(false);
  const [failed, setFailed] = useState(false);
  const plot = PLOT_TITLES[result?.problem_type ?? "regression"] ?? PLOT_TITLES.regression;
  const src = runId ? plotUrl(runId) : "";

  return (
    <Panel
      title={result ? plot.title : "Evaluation plot"}
      caption={result ? plot.caption : "How the champion does on held-out data"}
      actions={
        result && !failed ? (
          <Button variant="ghost" size="icon-sm" aria-label="Zoom the plot" onClick={() => setOpen(true)} data-export-ignore className="size-9 lg:size-7">
            <ZoomIn />
          </Button>
        ) : undefined
      }
      data-export-span={exportSpan}
      className="@5xl/report:col-span-4"
    >
      {result && runId ? (
        failed ? (
          <PanelEmpty icon={<ScanSearch />} title="No plot for this run" description="The run finished without saving an evaluation plot." />
        ) : (
          <>
            <button
              type="button"
              onClick={() => setOpen(true)}
              aria-label="Open the plot larger"
              className="relative flex aspect-square w-full flex-1 cursor-zoom-in items-center justify-center overflow-hidden rounded-xl bg-background/40 ring-1 ring-border outline-none focus-visible:ring-3 focus-visible:ring-ring/50 lock:aspect-auto lock:min-h-0"
            >
              {/* eslint-disable-next-line @next/next/no-img-element -- backend-generated PNG */}
              <img src={src} alt={`${plot.title} for ${result.model_name ?? "the champion model"}`} onError={() => setFailed(true)} className="h-full w-full object-contain" />
            </button>
            <Dialog open={open} onOpenChange={setOpen}>
              <DialogContent className="sm:max-w-4xl">
                <DialogHeader>
                  <DialogTitle>{plot.title}</DialogTitle>
                  <DialogDescription>{plot.caption}</DialogDescription>
                </DialogHeader>
                {/* eslint-disable-next-line @next/next/no-img-element -- backend-generated PNG */}
                <img src={src} alt={plot.title} className="max-h-[75vh] w-full rounded-lg object-contain" />
              </DialogContent>
            </Dialog>
          </>
        )
      ) : (
        empty
      )}
    </Panel>
  );
}

/* ─── Feature importance ─────────────────────────────────────────────── */

function FeaturesPanel({ result, empty, exportSpan }: { result: RunResult | null; empty: React.ReactNode; exportSpan: number }) {
  const features = (result?.extra?.top_features ?? []).slice(0, 10);
  const max = Math.max(0, ...features.map((f) => f.importance));

  return (
    <Panel
      title="Feature importance"
      caption={features.length ? `Top ${features.length} features by importance` : "Which columns drive the prediction"}
      data-export-span={exportSpan}
      className="@5xl/report:col-span-3"
    >
      {!result ? (
        empty
      ) : features.length ? (
        <ol tabIndex={0} aria-label="Feature importance" className="-me-1 flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto rounded-lg pe-1 outline-none focus-visible:ring-3 focus-visible:ring-ring/50">
          {features.map((f) => (
            <li key={f.feature} className="min-w-0">
              <div className="mb-1 flex items-baseline justify-between gap-2 text-sm">
                <span className="min-w-0 truncate font-mono" title={f.feature}>
                  {f.feature}
                </span>
                <span className="shrink-0 font-mono text-xs text-muted-foreground tabular-nums">{f.importance.toFixed(4)}</span>
              </div>
              <Meter value={max > 0 ? (f.importance / max) * 100 : 0} className="h-2" barClassName="bg-chart-1" />
            </li>
          ))}
        </ol>
      ) : (
        <PanelEmpty icon={<SlidersHorizontal />} title="Not reported" description="This kind of model does not report feature importance." />
      )}
    </Panel>
  );
}

/* ─── Model selection: the leaderboard, then tuning of the champion ──── */

function ModelsPanel({ result, empty }: { result: RunResult | null; empty: React.ReactNode }) {
  const [view, setView] = useState<"leaderboard" | "tuning">("leaderboard");
  const metric = result?.score_metric ?? "score";
  const info = metricInfo(metric);
  const models = result?.extra?.all_models ?? [];
  const tuned = (result?.extra?.tuning_trials?.length ?? 0) > 1;
  const tuning = result && tuned ? tuningSummary(result) : null;

  return (
    <Panel
      title="Model selection"
      caption={
        result
          ? models.length
            ? `${models.length} candidates ranked by cross-validated ${info.short}${tuning ? `, then ${tuning.searched} tuning trials` : ""}`
            : "Only the champion was trained"
          : "Every candidate ranked, then the champion tuned"
      }
      data-export-span="12"
      className="@3xl/report:col-span-2 @5xl/report:col-span-5"
    >
      {!result ? (
        empty
      ) : (
        <>
          {tuning && (
            <div data-export-ignore className="shrink-0">
              <Segmented
                label="Model selection view"
                value={view}
                onChange={setView}
                options={[
                  { value: "leaderboard", label: "Leaderboard" },
                  { value: "tuning", label: "Tuning" },
                ]}
              />
            </div>
          )}
          {/* Both views are rendered; the export shows the hidden one too. */}
          <section hidden={view !== "leaderboard"} data-export-show className="flex min-h-0 flex-1 flex-col gap-2">
            {tuning && (
              <h3 hidden data-export-show className="text-xs font-medium text-muted-foreground">
                Leaderboard
              </h3>
            )}
            <Leaderboard result={result} />
          </section>
          {tuning && (
            <section hidden={view !== "tuning"} data-export-show className="flex min-h-0 flex-1 flex-col gap-3">
              <h3 hidden data-export-show className="text-xs font-medium text-muted-foreground">
                Hyperparameter tuning: Bayesian search (Optuna) on {result.model_name ?? "the champion"}
              </h3>
              <Tuning summary={tuning} />
            </section>
          )}
        </>
      )}
    </Panel>
  );
}

/* ─── Leaderboard and tuning views ──────────────────────────────────────── */

function parseParams(raw: string): [string, string][] | null {
  try {
    return Object.entries(JSON.parse(raw) as Record<string, unknown>).map(([k, v]) => [k, String(v)]);
  } catch {
    return null;
  }
}

// Keep the verdict and trim the LLM's reason so rows stay one line; the full
// text is in the title.
function shortOutcome(t: TuningTrial) {
  if (t.trial === 0) return "Baseline: champion from the initial sweep";
  const at = t.result.indexOf("Reason:");
  if (at === -1) return t.result;
  return `${t.result.slice(0, at).trim()} ${t.result.slice(at + 7).trim()}`;
}

function Leaderboard({ result }: { result: RunResult }) {
  const metric = result.score_metric ?? "score";
  const info = metricInfo(metric);
  const lowerIsBetter = result.higher_is_better === false || !!info.lowerIsBetter;
  const models = [...(result.extra?.all_models ?? [])].sort((a, b) => (lowerIsBetter ? a.cv_mean - b.cv_mean : b.cv_mean - a.cv_mean));
  const champion = result.model_name;

  if (!models.length) {
    return <PanelEmpty icon={<ListOrdered />} title="Single model" description="This run trained one model, so there is nothing to rank." />;
  }
  return (
    <div
      tabIndex={0}
      role="region"
      aria-label="Model leaderboard"
      className="min-h-0 flex-1 overflow-auto rounded-xl bg-background/30 ring-1 ring-border outline-none focus-visible:ring-3 focus-visible:ring-ring/50 *:data-[slot=table-container]:overflow-visible"
    >
      <Table className="text-xs">
        <TableHeader className="sticky top-0 z-10 bg-popover backdrop-blur-xl">
          <TableRow className="hover:bg-transparent">
            <TableHead className="h-9 w-10 px-3 text-end text-xs text-muted-foreground">#</TableHead>
            <TableHead className="h-9 px-3 text-xs text-muted-foreground">Model</TableHead>
            <TableHead className="h-9 px-3 text-end text-xs text-muted-foreground">CV mean</TableHead>
            <TableHead className="h-9 px-3 text-end text-xs text-muted-foreground">CV std</TableHead>
            <TableHead className="h-9 px-3 text-end text-xs text-muted-foreground">Test</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody className="font-mono">
          {models.map((m, i) => {
            const isChampion = m.name === champion;
            const test = isChampion ? result.extra?.test_score : undefined;
            return (
              <TableRow key={m.name} className={cn("border-border/60", isChampion && "bg-primary/10 text-primary")}>
                <TableCell className="px-3 py-1.5 text-end tabular-nums">{i + 1}</TableCell>
                <TableCell className="max-w-48 truncate px-3 py-1.5 font-medium" title={m.name}>
                  <span className="inline-flex items-center gap-1.5">
                    {isChampion && <Trophy aria-label="Champion" className="size-3.5 shrink-0" />}
                    {m.name}
                  </span>
                </TableCell>
                <TableCell className="px-3 py-1.5 text-end tabular-nums">{formatScore(m.cv_mean, metric)}</TableCell>
                <TableCell className="px-3 py-1.5 text-end text-muted-foreground tabular-nums">±{m.cv_std.toFixed(4)}</TableCell>
                <TableCell className="px-3 py-1.5 text-end tabular-nums">{test !== undefined ? formatScore(test, metric, 1) : "-"}</TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

function tuningSummary(result: RunResult) {
  const trials = result.extra?.tuning_trials ?? [];
  const metric = result.score_metric ?? "score";
  const info = metricInfo(metric);
  const lowerIsBetter = result.higher_is_better === false || !!info.lowerIsBetter;
  const baseline = trials[0]?.score ?? result.accuracy_score ?? 0;
  const optimized = result.accuracy_score ?? baseline;
  const delta = optimized - baseline;
  const improved = lowerIsBetter ? delta < -0.0001 : delta > 0.0001;
  const deltaText = info.percent ? `${delta >= 0 ? "+" : ""}${(delta * 100).toFixed(2)} pts` : `${delta >= 0 ? "+" : ""}${delta.toFixed(4)}`;
  return { trials, metric, baseline, optimized, improved, deltaText, searched: trials.filter((t) => t.trial > 0).length };
}

function Tuning({ summary }: { summary: ReturnType<typeof tuningSummary> }) {
  const { trials, metric, baseline, optimized, improved } = summary;
  // The trial that set the final score; later ties only matched it.
  const bestTrial = improved
    ? trials.find((t) => t.trial > 0 && t.score !== null && Math.abs(t.score - optimized) < 0.00005)?.trial
    : undefined;
  return (
    <>
      <dl className="grid grid-cols-2 gap-3">
        <div className="rounded-xl bg-muted/60 px-3 py-2 ring-1 ring-border lock-short:py-1">
          <dt className="text-xs text-muted-foreground">Before, default params</dt>
          <dd className="font-mono text-xl font-bold text-muted-foreground tabular-nums">{formatScore(baseline, metric)}</dd>
        </div>
        <div className={cn("rounded-xl px-3 py-2 ring-1 lock-short:py-1", improved ? "bg-green-500/10 ring-green-400/25" : "bg-muted/60 ring-border")}>
          <dt className="text-xs text-muted-foreground">After, best params</dt>
          <dd className={cn("font-mono text-xl font-bold tabular-nums", improved && "text-green-300")}>{formatScore(optimized, metric)}</dd>
        </div>
      </dl>

      <ol tabIndex={0} aria-label="Tuning trials" className="-me-1 flex min-h-0 flex-1 flex-col divide-y divide-border/60 overflow-y-auto rounded-lg pe-1 outline-none focus-visible:ring-3 focus-visible:ring-ring/50">
        {trials.map((t) => {
          const params = t.trial === 0 ? null : parseParams(t.parameters);
          const isBest = t.trial === bestTrial;
          const lower = t.result.toLowerCase();
          return (
            <li key={t.trial} className={cn("min-w-0 py-1.5", isBest && "-mx-2 rounded-lg bg-green-500/10 px-2")}>
              <div className="flex min-w-0 items-baseline gap-2 text-xs">
                <span className={cn("w-8 shrink-0 font-mono font-bold tabular-nums", isBest ? "text-green-300" : t.trial === 0 ? "text-primary" : "text-muted-foreground")}>
                  #{t.trial}
                </span>
                <span className={cn("min-w-0 flex-1 truncate", lower.includes("error") && "text-red-300", lower.includes("new champion") && "font-medium text-green-300")} title={t.result}>
                  {formatArabicBrand(shortOutcome(t))}
                </span>
                <span className="shrink-0 font-mono font-bold tabular-nums">{t.score === null ? "-" : formatScore(t.score, metric)}</span>
              </div>
              {t.trial > 0 && (
                <p className="ms-10 truncate font-mono text-[0.6875rem] text-muted-foreground" title={t.parameters} dir="ltr">
                  {params ? params.map(([k, v]) => `${k}=${v}`).join("  ") : t.parameters}
                </p>
              )}
            </li>
          );
        })}
      </ol>
    </>
  );
}
