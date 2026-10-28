"use client";

import { useEffect, useRef, useState } from "react";
import {
  ArrowRight,
  ChartColumn,
  ChartLine,
  Check,
  ChevronDown,
  CircleAlert,
  Code,
  Cog,
  Database,
  Gauge,
  RotateCcw,
  Square,
  SquareTerminal,
  Target,
  Workflow,
  Wrench,
} from "lucide-react";
import { cancelRun } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Fact, Meter, Panel, PanelEmpty, PanelError, PanelSkeleton } from "@/components/panel";
import { toast } from "@/components/ui/toast";
import { Segmented } from "@/components/segmented";
import { STAGE_TO_STEP, STEPS, type PipelineStep } from "./pipeline-steps";
import { hasStarted, isRunActive, isRunFailed, type RunState } from "./use-run";
import type { StageKey } from "./dashboard";

const STEP_ICONS = [Database, Target, ChartColumn, Wrench, Cog, ChartLine];

type StepStatus = "waiting" | "active" | "done" | "error";
type LogFilter = "all" | "success" | "system";

const humanize = (s?: string | null) => (s ? s.replace(/_/g, " ") : "-");

// Older backends report the GPU before it starts as "NVIDIA H200 (requested)".
const gpuName = (gpu?: string) => gpu?.replace(/\s*\(requested\)\s*$/i, "") || "-";

function formatElapsed(seconds: number | null) {
  if (seconds == null) return "-";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = String(seconds % 60).padStart(2, "0");
  return h ? `${h}h ${String(m).padStart(2, "0")}m ${s}s` : m ? `${m}m ${s}s` : `${seconds}s`;
}

// How far one tick may pull the clock forward to meet an earlier start.
const CLOCK_SLEW_MS = 250;

/**
 * The run's elapsed time as a clock that only counts up. Each diagnostics poll
 * reports whole seconds since the run was queued, which puts the start at the
 * moment the report arrived minus that many seconds. Network delay only ever
 * makes a report look later, so the earliest start seen is the best one, and a
 * second API container whose clock runs a little ahead cannot drag the time
 * back. The clock ticks on this device's own time, on each second boundary, so
 * it never skips or repeats a second; an earlier start found later is taken up
 * a quarter second per tick. Once the run stops, the last value stays.
 */
function useRunClock(reported: number | null | undefined, live: boolean): number | null {
  const start = useRef<number | null>(null);
  const best = useRef<number | null>(null);
  const [seconds, setSeconds] = useState<number | null>(null);

  useEffect(() => {
    if (reported == null) return;
    const estimate = Date.now() - reported * 1000;
    if (best.current === null || estimate < best.current) best.current = estimate;
    start.current ??= best.current;
  }, [reported]);

  useEffect(() => {
    if (!live) return;
    let timer: ReturnType<typeof setTimeout>;
    const tick = () => {
      if (start.current !== null && best.current !== null && best.current < start.current) {
        start.current = Math.max(best.current, start.current - CLOCK_SLEW_MS);
      }
      const since = start.current === null ? null : Date.now() - start.current;
      if (since !== null) setSeconds((prev) => Math.max(prev ?? 0, Math.floor(since / 1000)));
      timer = setTimeout(tick, since === null ? 250 : 1000 - (since % 1000) + 20);
    };
    tick();
    return () => clearTimeout(timer);
  }, [live]);

  return seconds;
}

export function TrainingStage({
  runId,
  run,
  onNewRun,
  goToStage,
}: {
  runId: string | null;
  run: RunState;
  onNewRun: () => void;
  goToStage: (key: StageKey) => void;
}) {
  const { status, diagnostics, setStatus, statusError, notFound, reload } = run;
  const phase = status?.status;
  const loading = !!runId && !status && !statusError;
  const started = hasStarted(status);
  const active = isRunActive(status);
  const failed = isRunFailed(status);
  const succeeded = phase === "succeeded";

  // Which cell is running comes from the backend's stage, which also survives
  // a failure, so the failed cell is the one that was running.
  const stageStep = status?.stage ? STAGE_TO_STEP[status.stage] : undefined;
  const currentStep = succeeded ? STEPS.length : phase === "running" || failed ? (stageStep ?? 0) : -1;

  const stepStatus = (i: number): StepStatus =>
    i < currentStep ? "done" : i === currentStep ? (failed ? "error" : "active") : "waiting";

  // Per-cell durations, timed in this session as the stage advances.
  const [stepSeconds, setStepSeconds] = useState<Record<number, number>>({});
  const stepStartedAt = useRef<number | null>(null);
  const prevStep = useRef(-1);
  useEffect(() => {
    const now = Date.now();
    const prev = prevStep.current;
    if (prev >= 0 && currentStep > prev && stepStartedAt.current !== null) {
      const secs = Math.max(1, Math.round((now - stepStartedAt.current) / 1000));
      setStepSeconds((m) => ({ ...m, [prev]: secs }));
    }
    if (currentStep !== prev) {
      stepStartedAt.current = now;
      prevStep.current = currentStep;
    }
  }, [currentStep]);

  // The server's elapsed time counts from when the run was queued and keeps
  // counting after the run ends, so a finished run shows the last value seen
  // while it was live, or nothing.
  const elapsed = useRunClock(diagnostics?.elapsed_seconds, active);

  const doneCount = Math.max(0, Math.min(currentStep, STEPS.length));
  const goToData = (
    <Button variant="outline" onClick={() => goToStage("data")} className="h-10 lg:h-8">
      Go to Data
    </Button>
  );

  // Locked: the pipeline above the run's facts on the left, the notebook in
  // the rest. On phones the facts and the cancel button come before the long
  // notebook.
  return (
    <div className="grid flex-1 gap-3 md:grid-cols-2 lock:h-full lock:grid-cols-12 lock:grid-rows-[minmax(0,1fr)_auto]">
      {/* Pipeline */}
      <Panel
        title="Pipeline"
        caption={
          started
            ? succeeded
              ? `All ${STEPS.length} stages complete`
              : failed
                ? `Stopped at stage ${Math.min(currentStep + 1, STEPS.length)} of ${STEPS.length}`
                : phase === "queued"
                  ? "Waiting for a GPU container"
                  : `Stage ${Math.min(currentStep + 1, STEPS.length)} of ${STEPS.length}`
            : "Profile, detect, explore, engineer, train, report"
        }
        className="lock:col-span-4"
      >
        {loading ? (
          <PanelSkeleton rows={6} />
        ) : statusError && !status ? (
          <PanelError
            title={notFound ? "Run not found" : "Couldn't load this run"}
            message={notFound ? "This run doesn't exist on the server." : statusError}
            onRetry={notFound ? undefined : reload}
            action={
              <Button variant={notFound ? "default" : "ghost"} onClick={onNewRun} className="h-10 lg:h-8">
                Start a new run
              </Button>
            }
          />
        ) : !started ? (
          <PanelEmpty
            icon={<Workflow />}
            title="No run yet"
            description="Configure a run on the Data stage and start training."
            action={goToData}
          />
        ) : (
          <>
            <div className="flex items-center gap-3">
              <Meter
                value={(doneCount / STEPS.length) * 100}
                className="flex-1"
                barClassName={failed ? "bg-red-400" : succeeded ? "bg-green-400" : undefined}
              />
              <span className="font-mono text-xs text-muted-foreground tabular-nums">
                {doneCount}/{STEPS.length}
              </span>
            </div>

            {/* On a locked page the list is a size container: each step's summary shows only where all six fit with theirs. */}
            <ol tabIndex={0} aria-label="Pipeline stages" className="-mx-1 flex min-h-0 flex-1 flex-col gap-0.5 overflow-y-auto rounded-lg px-1 outline-none focus-visible:ring-3 focus-visible:ring-ring/50 lock:[container:steps/size]">
              {STEPS.map((step, i) => {
                const st = stepStatus(i);
                const StepIcon = STEP_ICONS[i];
                return (
                  <li
                    key={step.key}
                    className={cn(
                      "flex items-center gap-3 rounded-lg px-2 py-1.5 lock-short:py-1",
                      st === "active" && "bg-primary/10",
                      st === "error" && "bg-red-500/10"
                    )}
                  >
                    <span
                      className={cn(
                        "flex size-7 shrink-0 items-center justify-center rounded-full ring-1 lock-short:size-6 [&_svg]:size-3.5",
                        st === "done" && "bg-green-500/15 text-green-300 ring-green-400/35",
                        st === "active" && "bg-primary/15 text-primary ring-primary/45",
                        st === "error" && "bg-red-500/15 text-red-300 ring-red-400/40",
                        st === "waiting" && "bg-muted text-muted-foreground ring-border"
                      )}
                    >
                      {st === "done" ? (
                        <Check />
                      ) : st === "error" ? (
                        <CircleAlert />
                      ) : st === "active" ? (
                        <StepIcon className="motion-safe:animate-pulse" />
                      ) : (
                        <span className="font-mono text-[0.6875rem] tabular-nums">{i + 1}</span>
                      )}
                    </span>
                    <div className="min-w-0 flex-1">
                      <p
                        className={cn(
                          "truncate text-sm",
                          st === "waiting" && "text-muted-foreground",
                          st === "active" && "font-medium text-primary",
                          st === "error" && "font-medium text-red-200"
                        )}
                      >
                        {step.label}
                      </p>
                      <p className="text-xs text-muted-foreground lock:truncate steps-tight:hidden" title={step.summary}>
                        {step.summary}
                      </p>
                    </div>
                    {st === "done" && stepSeconds[i] != null && (
                      <span className="font-mono text-xs text-muted-foreground tabular-nums">{stepSeconds[i]}s</span>
                    )}
                  </li>
                );
              })}
            </ol>

            {succeeded && (
              <div className="flex flex-wrap items-center gap-3 rounded-xl bg-green-500/10 p-3 ring-1 ring-green-400/25">
                <p className="min-w-0 flex-1 text-sm lock-short:hidden">Champion selected and metrics ready.</p>
                <Button onClick={() => goToStage("results")} className="h-10 lock-short:flex-1 lg:h-8">
                  View results
                  <ArrowRight data-icon="inline-end" />
                </Button>
              </div>
            )}
            {failed && (
              <div className="flex flex-col gap-2 rounded-xl bg-red-500/10 p-3 ring-1 ring-red-400/25">
                <p className="text-sm font-medium text-red-200">{phase === "cancelled" ? "Run cancelled" : "Run failed"}</p>
                {status?.error && (
                  <p className="line-clamp-2 text-xs text-muted-foreground" title={status.error}>
                    {status.error}
                  </p>
                )}
                <Button variant="outline" size="sm" onClick={onNewRun} className="h-9 self-start lg:h-7">
                  <RotateCcw data-icon="inline-start" />
                  Start over
                </Button>
              </div>
            )}
          </>
        )}
      </Panel>

      {/* Run diagnostics: the clock first, then what the run is and runs on. */}
      <Panel
        title="Run diagnostics"
        caption="Recorded run state. Training runs on separate GPU containers."
        actions={active && runId ? <CancelRun runId={runId} onCancelled={(s) => s?.status && setStatus(s)} /> : undefined}
        className="lock:col-span-4 lock:row-start-2"
      >
        {loading ? (
          <PanelSkeleton rows={2} />
        ) : !started ? (
          <PanelEmpty icon={<Gauge />} title="Nothing recorded" description="Stage, hardware and timing show here while a run trains." />
        ) : (
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3 lock-short:gap-y-2">
            {(active || elapsed != null) && (
              <Fact
                label="Elapsed"
                value={formatElapsed(elapsed)}
                className="col-span-full lock-short:col-span-1"
                valueClassName="font-mono text-2xl leading-tight tabular-nums lock-short:text-sm lock-short:leading-snug"
              />
            )}
            <Fact label="Stage" value={humanize(diagnostics?.stage ?? status?.stage) || "waiting"} className="capitalize" />
            <Fact label="Task" value={humanize(diagnostics?.task ?? status?.task)} />
            <Fact label="Target" value={status?.target ?? "-"} title={status?.target ?? undefined} valueClassName="font-mono" />
            <Fact label="Training on" value={gpuName(diagnostics?.gpu)} title={gpuName(diagnostics?.gpu)} />
            <Fact label="GPU containers" value={diagnostics ? diagnostics.gpu_calls : "-"} valueClassName="font-mono tabular-nums" />
          </dl>
        )}
      </Panel>

      {/* Notebook */}
      <NotebookPanel
        runId={runId}
        started={started}
        loading={loading}
        stepStatus={stepStatus}
        currentStep={currentStep}
        stepSeconds={stepSeconds}
        error={failed ? (status?.error ?? (phase === "cancelled" ? "Cancelled." : "Stage failed.")) : null}
        goToData={goToData}
        className="md:col-span-2 lock:col-span-8 lock:col-start-5 lock:row-span-2 lock:row-start-1"
      />
    </div>
  );
}

function CancelRun({ runId, onCancelled }: { runId: string; onCancelled: (s: Awaited<ReturnType<typeof cancelRun>>) => void }) {
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const cancel = async () => {
    setBusy(true);
    setError(null);
    try {
      onCancelled(await cancelRun(runId));
      setOpen(false);
      toast.add({ title: "Run cancelled", description: "Training stopped. Start a new run whenever you're ready.", type: "info" });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not cancel the run.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <Button variant="destructive" size="sm" onClick={() => setOpen(true)} className="h-9 lg:h-7">
        <Square data-icon="inline-start" />
        Cancel run
      </Button>
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Cancel this run?</DialogTitle>
            <DialogDescription>Training stops and the run cannot be resumed. You would need to start a new run.</DialogDescription>
          </DialogHeader>
          {error && (
            <p role="alert" className="text-xs text-red-300">
              {error}
            </p>
          )}
          <DialogFooter>
            <DialogClose render={<Button variant="outline" />}>Keep training</DialogClose>
            <Button variant="destructive" onClick={cancel} disabled={busy}>
              {busy ? "Cancelling..." : "Cancel run"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}

/* ─── Notebook ───────────────────────────────────────────────────────── */

function NotebookPanel({
  runId,
  started,
  loading,
  stepStatus,
  currentStep,
  stepSeconds,
  error,
  goToData,
  className,
}: {
  runId: string | null;
  started: boolean;
  loading: boolean;
  stepStatus: (i: number) => StepStatus;
  currentStep: number;
  stepSeconds: Record<number, number>;
  error: string | null;
  goToData: React.ReactNode;
  className?: string;
}) {
  const [filter, setFilter] = useState<LogFilter>("all");
  const [expanded, setExpanded] = useState<Set<number>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);
  const cellRefs = useRef<(HTMLElement | null)[]>([]);

  // Follow the running cell: open it, fold the one before, and bring it into
  // view inside the notebook only (scrollIntoView would also slide the track).
  useEffect(() => {
    if (currentStep < 0 || currentStep >= STEPS.length) return;
    setExpanded((prev) => {
      const next = new Set(prev);
      next.add(currentStep);
      next.delete(currentStep - 1);
      return next;
    });
    const box = scrollRef.current;
    const cell = cellRefs.current[currentStep];
    if (box && cell) box.scrollTo({ top: Math.max(0, cell.offsetTop - 8), behavior: "smooth" });
  }, [currentStep]);

  const toggle = (i: number) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(i)) next.delete(i);
      else next.add(i);
      return next;
    });

  return (
    <Panel
      title="Notebook"
      caption={runId ? `automl_forge_${runId.slice(0, 8)}.ipynb` : "One cell per pipeline stage"}
      actions={
        started ? (
          <Segmented
            label="Log level"
            value={filter}
            onChange={setFilter}
            options={[
              { value: "all", label: "All" },
              { value: "success", label: "Success" },
              { value: "system", label: "System" },
            ]}
          />
        ) : undefined
      }
      className={className}
    >
      {loading ? (
        <PanelSkeleton rows={6} />
      ) : !started ? (
        <PanelEmpty
          icon={<SquareTerminal />}
          title="The notebook is empty"
          description="Each pipeline stage runs as a cell here, with its code and log."
          action={goToData}
        />
      ) : (
        <div ref={scrollRef} className="relative -mx-1 flex max-h-[34rem] min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-1 lock:max-h-none">
          {STEPS.map((step, i) => (
            <NotebookCell
              key={step.key}
              ref={(el) => {
                cellRefs.current[i] = el;
              }}
              step={step}
              number={i + 1}
              status={stepStatus(i)}
              seconds={stepSeconds[i]}
              error={error}
              expanded={expanded.has(i)}
              onToggle={() => toggle(i)}
              filter={filter}
            />
          ))}
        </div>
      )}
    </Panel>
  );
}

const LINE_KIND = {
  SUCCESS: { mark: "✓", className: "text-green-300" },
  SYSTEM: { mark: "⚙", className: "text-primary" },
  INFO: { mark: "i", className: "text-chart-3" },
  ERROR: { mark: "!", className: "text-red-300" },
} as const;

function NotebookCell({
  ref,
  step,
  number,
  status,
  seconds,
  error,
  expanded,
  onToggle,
  filter,
}: {
  ref: React.Ref<HTMLElement>;
  step: PipelineStep;
  number: number;
  status: StepStatus;
  seconds?: number;
  error: string | null;
  expanded: boolean;
  onToggle: () => void;
  filter: LogFilter;
}) {
  const [visible, setVisible] = useState(0);

  // Reveal the running cell's log a line at a time. The closing SUCCESS line is
  // held back until the stage has actually finished.
  useEffect(() => {
    if (status !== "active") {
      setVisible(0);
      return;
    }
    let i = 0;
    let timer: ReturnType<typeof setTimeout>;
    const tick = () => {
      i += 1;
      setVisible(i);
      if (i < step.outputLines.length - 1) timer = setTimeout(tick, 350 + Math.random() * 250);
    };
    timer = setTimeout(tick, 200);
    return () => clearTimeout(timer);
  }, [status, step.outputLines.length]);

  const finished = status === "done" || status === "error";
  // A failed cell never reached its SUCCESS line; it ends on the error instead.
  const source =
    status === "active"
      ? step.outputLines.slice(0, visible)
      : status === "error"
        ? [...step.outputLines.slice(0, -1), `[ERROR] ${error ?? "Stage failed."}`]
        : step.outputLines;
  const lines = source.filter((line) =>
    filter === "all" ? true : filter === "success" ? line.startsWith("[SUCCESS]") : line.startsWith("[SYSTEM]")
  );
  const showOutput = status === "active" || (finished && expanded);
  const [fnName, fnArgs = ""] = step.fn.split("(");
  const counter = status === "waiting" ? " " : status === "active" ? "*" : status === "done" ? String(number) : "!";

  return (
    <Collapsible
      render={<article ref={ref} />}
      className={cn(
        "shrink-0 overflow-hidden rounded-xl bg-background/40 ring-1 transition-colors",
        status === "active" && "ring-primary/40",
        status === "done" && "ring-green-400/25",
        status === "error" && "ring-red-400/40",
        status === "waiting" && "bg-background/20 ring-border"
      )}
    >
      <div className="flex min-w-0 items-center gap-2 px-3 py-2">
        <span
          className={cn(
            "shrink-0 font-mono text-xs whitespace-nowrap",
            status === "active" && "text-primary",
            status === "done" && "text-green-300",
            status === "error" && "text-red-300",
            status === "waiting" && "text-muted-foreground"
          )}
          dir="ltr"
        >
          In [{counter}]:
        </span>
        <code className="min-w-0 flex-1 truncate font-mono text-xs" dir="ltr" title={step.fn}>
          <span className={cn("font-bold", status === "waiting" ? "text-muted-foreground" : "text-primary")}>{fnName}</span>
          <span className="text-muted-foreground">({fnArgs}</span>
        </code>
        {status === "done" && seconds != null && (
          <span className="font-mono text-xs text-muted-foreground tabular-nums">{seconds}s</span>
        )}
        {status !== "waiting" && (
          <CollapsibleTrigger render={<Button variant="ghost" size="xs" className="h-8 lg:h-6" />} aria-label={`Code for ${fnName}`}>
            <Code data-icon="inline-start" />
            <span className="hidden sm:inline">Code</span>
          </CollapsibleTrigger>
        )}
        {finished && (
          <Button
            variant="ghost"
            size="icon-xs"
            className="size-8 lg:size-6"
            aria-label={expanded ? "Hide output" : "Show output"}
            aria-expanded={expanded}
            onClick={onToggle}
          >
            <ChevronDown className={cn("transition-transform", expanded && "rotate-180")} />
          </Button>
        )}
      </div>

      <CollapsibleContent>
        <pre className="overflow-x-auto border-t border-border bg-background/60 px-3 py-2.5 font-mono text-xs leading-relaxed text-foreground/85" dir="ltr">
          {step.code}
        </pre>
      </CollapsibleContent>

      {showOutput && (
        <div className="space-y-0.5 border-t border-border px-3 py-2 font-mono text-xs leading-relaxed" dir="ltr">
          {lines.length ? (
            lines.map((line, i) => {
              const kind = (line.match(/^\[([A-Z]+)\]/)?.[1] ?? "INFO") as keyof typeof LINE_KIND;
              const k = LINE_KIND[kind] ?? LINE_KIND.INFO;
              return (
                <p key={i} className="text-muted-foreground">
                  <span className={cn("me-2 font-bold", k.className)}>
                    {k.mark} [{kind}]
                  </span>
                  {line.replace(/^\[[A-Z]+\]\s*/, "")}
                  {status === "active" && i === lines.length - 1 && (
                    <span aria-hidden="true" className="ms-1 inline-block h-3 w-1.5 translate-y-0.5 bg-primary motion-safe:animate-pulse" />
                  )}
                </p>
              );
            })
          ) : (
            <p className="py-1 text-center text-muted-foreground">No log lines at this level.</p>
          )}
        </div>
      )}
    </Collapsible>
  );
}
