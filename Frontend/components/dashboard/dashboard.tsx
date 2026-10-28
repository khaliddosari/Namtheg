"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Plus } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { SiteHeader } from "@/components/site-header";
import type { Tone } from "@/components/panel";
import type { RunStatus } from "@/lib/api";
import { isRunActive, isRunFailed, isUploadFailed, landingStage, useRun } from "./use-run";
import { AccountControl } from "./account";
import { Pager } from "./pager";
import { IntroStage } from "./intro-stage";
import { DataStage } from "./data-stage";
import { TrainingStage } from "./training-stage";
import { ResultsStage } from "./results-stage";
import { DeployStage } from "./deploy-stage";

export const STAGES = [
  { key: "data", label: "Data" },
  { key: "training", label: "Training" },
  { key: "results", label: "Results" },
  { key: "deploy", label: "Deploy" },
] as const;

export type StageKey = (typeof STAGES)[number]["key"];

// The track: the intro poster first, then the four stages.
const SLIDES = [{ key: "intro", label: "Intro" }, ...STAGES] as const;

const slideIndex = (key: string | null) => SLIDES.findIndex((s) => s.key === key);
const DATA = slideIndex("data");

// Arrow keys page between stages unless something focused uses them itself.
const OWNS_ARROW_KEYS =
  'input, textarea, select, [contenteditable="true"], [role="slider"], [role="listbox"], [role="option"], [role="menu"], [role="tablist"], [role="radiogroup"], [data-slot="toggle-group"], [data-slot="dialog-content"]';

type HistoryMode = "push" | "replace" | "none";

/**
 * Points the address bar at a run and slide without navigating. The bare
 * address is the intro; every other slide names itself, run or not, so a
 * reload stays put.
 */
function writeUrl(runId: string | null, slide: number, mode: Exclude<HistoryMode, "none">) {
  const url = new URL(window.location.href);
  const key = SLIDES[slide].key;
  if (runId) url.searchParams.set("run", runId);
  else url.searchParams.delete("run");
  if (runId || key !== "intro") url.searchParams.set("stage", key);
  else url.searchParams.delete("stage");
  if (url.href === window.location.href) return;
  if (mode === "push") window.history.pushState(null, "", url);
  else window.history.replaceState(null, "", url);
}

/**
 * The whole app on one page. An intro poster comes first, then each stage is a
 * page of cards; the slides sit side by side in a horizontally snapping track,
 * so moving between them slides the track instead of loading a route. Every
 * slide stays mounted, which keeps form state and polling alive while you look
 * at another one. Each move is a history entry, so the browser's back and
 * forward buttons step through them.
 */
export function Dashboard() {
  const [runId, setRunId] = useState<string | null>(null);
  const [session, setSession] = useState(0);
  const [active, setActive] = useState(0);
  const [ready, setReady] = useState(false);
  const run = useRun(runId);
  const { setStatus } = run;

  const trackRef = useRef<HTMLDivElement>(null);
  const sectionRefs = useRef<(HTMLElement | null)[]>([]);
  const activeRef = useRef(0);
  const runIdRef = useRef<string | null>(null);
  const focusStage = useRef<number | null>(null);
  // The stage a smooth programmatic scroll is still travelling to. Until it
  // arrives, positions along the way are not a place the user settled on.
  const heading = useRef<{ stage: number; until: number } | null>(null);
  // The run whose landing stage has been decided, so a poll never yanks the view.
  const landedFor = useRef<string | null>(null);

  const goTo = useCallback((index: number, opts: { instant?: boolean; history?: HistoryMode; focus?: boolean } = {}) => {
    const i = Math.max(0, Math.min(SLIDES.length - 1, index));
    const moved = i !== activeRef.current;
    activeRef.current = i;
    setActive(i);
    if (opts.focus) focusStage.current = i;
    const history = opts.history ?? "push";
    if (history !== "none") writeUrl(runIdRef.current, i, history === "push" && moved ? "push" : "replace");
    const track = trackRef.current;
    if (!track) return;
    const smooth = !opts.instant && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    heading.current = smooth ? { stage: i, until: Date.now() + 1500 } : null;
    track.scrollTo({ left: i * track.clientWidth, behavior: smooth ? "smooth" : "instant" });
  }, []);

  // Buttons inside a stage move focus into the stage they open; the one they
  // were in turns inert, which would otherwise drop focus on the floor.
  const goToStage = useCallback((key: StageKey) => goTo(slideIndex(key), { focus: true }), [goTo]);

  useEffect(() => {
    if (focusStage.current !== active) return;
    focusStage.current = null;
    sectionRefs.current[active]?.focus({ preventScroll: true });
  }, [active]);

  const adoptRun = useCallback((id: string | null) => {
    runIdRef.current = id;
    landedFor.current = id;
    setRunId(id);
    setSession((s) => s + 1);
  }, []);

  // Deep links: /?run=<id>&stage=<key>, where the key is intro or a stage. The
  // bare address opens the intro. The legacy /<id>/<page> routes redirect here.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const linkedRun = params.get("run");
    const linkedSlide = slideIndex(params.get("stage"));
    if (linkedRun) {
      runIdRef.current = linkedRun;
      setRunId(linkedRun);
    }
    if (linkedSlide >= 0) {
      landedFor.current = linkedRun;
      goTo(linkedSlide, { instant: true, history: "replace" });
    } else if (linkedRun) {
      // Until the run says where it has got to, it waits on Data, which every
      // run has reached and which says so if the run can't be loaded.
      goTo(DATA, { instant: true, history: "none" });
    }
    setReady(true);
  }, [goTo]);

  // Back and forward: follow the address bar to its run and slide.
  useEffect(() => {
    const onPop = () => {
      const params = new URLSearchParams(window.location.search);
      const linkedRun = params.get("run");
      if (linkedRun !== runIdRef.current) adoptRun(linkedRun);
      goTo(Math.max(0, slideIndex(params.get("stage"))), { history: "none" });
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, [adoptRun, goTo]);

  // A link with a run but no stage lands wherever that run is.
  useEffect(() => {
    if (!runId || !run.status || landedFor.current === runId) return;
    landedFor.current = runId;
    goTo(slideIndex(landingStage(run.status)), { instant: true, history: "replace" });
  }, [runId, run.status, goTo]);

  // Swipes and trackpad scrolls settle on a page; adopt it once they stop.
  // A resize re-snaps to the current page rather than landing between two.
  useEffect(() => {
    const track = trackRef.current;
    if (!track) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const onScroll = () => {
      clearTimeout(timer);
      timer = setTimeout(() => {
        const width = Math.max(1, track.clientWidth);
        const flight = heading.current;
        if (flight) {
          // Mid-flight pauses (a busy frame) are not a stop; wait for arrival.
          const arrived = Math.abs(track.scrollLeft - flight.stage * width) < 2;
          if (!arrived && Date.now() < flight.until) return;
          heading.current = null;
        }
        const i = Math.round(track.scrollLeft / width);
        if (i !== activeRef.current) goTo(i, { instant: true });
      }, 120);
    };
    // A swipe or wheel during a programmatic scroll takes over from it.
    const takeOver = () => {
      heading.current = null;
    };
    track.addEventListener("wheel", takeOver, { passive: true });
    track.addEventListener("touchstart", takeOver, { passive: true });
    const resize = new ResizeObserver(() => {
      track.scrollTo({ left: activeRef.current * track.clientWidth, behavior: "instant" });
    });
    track.addEventListener("scroll", onScroll, { passive: true });
    resize.observe(track);
    return () => {
      clearTimeout(timer);
      track.removeEventListener("scroll", onScroll);
      track.removeEventListener("wheel", takeOver);
      track.removeEventListener("touchstart", takeOver);
      resize.disconnect();
    };
  }, [goTo]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      if ((e.target as HTMLElement | null)?.closest?.(OWNS_ARROW_KEYS)) return;
      e.preventDefault();
      goTo(activeRef.current + (e.key === "ArrowRight" ? 1 : -1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [goTo]);

  const onUploaded = useCallback((id: string) => {
    runIdRef.current = id;
    landedFor.current = id;
    setRunId(id);
    writeUrl(id, activeRef.current, "push");
  }, []);

  const onStarted = useCallback(
    (status: RunStatus) => {
      setStatus(status);
      goToStage("training");
    },
    [setStatus, goToStage]
  );

  const newRun = useCallback(() => {
    writeUrl(null, DATA, "push");
    adoptRun(null);
    goTo(DATA, { history: "none" });
  }, [adoptRun, goTo]);

  // A run picked from the account opens wherever it has got to, as its link would.
  const openRun = useCallback(
    (id: string, status: RunStatus) => {
      const stage = slideIndex(landingStage(status));
      if (id === runIdRef.current) return goTo(stage);
      writeUrl(id, stage, "push");
      adoptRun(id);
      goTo(stage, { history: "none" });
    },
    [adoptRun, goTo]
  );

  const tones = stageTones(runId, run);

  return (
    <div className="flex h-dvh flex-col overflow-hidden">
      <h1 className="sr-only">Namtheg AutoML</h1>
      <SiteHeader
        onBrandClick={() => goTo(0)}
        start={
          <div className="flex min-w-0 items-center gap-2">
            <Button variant="outline" size="sm" onClick={newRun} aria-label="New run" className="size-9 px-0 sm:w-auto sm:ps-2 sm:pe-2.5 lg:h-8">
              <Plus />
              <span className="hidden sm:inline">New run</span>
            </Button>
            <AccountControl runId={runId} onOpenRun={openRun} />
          </div>
        }
      />

      <main className={cn("relative min-h-0 flex-1", !ready && "invisible")}>
        <div
          ref={trackRef}
          className="no-scrollbar flex h-full snap-x snap-mandatory overflow-x-auto overflow-y-hidden overscroll-x-contain"
        >
          {SLIDES.map((stage, i) => (
            <section
              key={stage.key}
              ref={(el) => {
                sectionRefs.current[i] = el;
              }}
              id={`stage-${stage.key}`}
              aria-label={stage.key === "intro" ? "Intro" : `${stage.label} stage`}
              tabIndex={-1}
              inert={i !== active}
              className="h-full w-full shrink-0 snap-start snap-always overflow-y-auto overscroll-y-contain outline-none @container-size/page"
            >
              <div className="mx-auto flex min-h-full w-full max-w-[120rem] flex-col p-3 sm:p-4 lock:h-full">
                {stage.key === "intro" && <IntroStage onStart={() => goToStage("data")} />}
                {stage.key === "data" && (
                  <DataStage
                    key={session}
                    runId={runId}
                    run={run}
                    onUploaded={onUploaded}
                    onStarted={onStarted}
                    onNewRun={newRun}
                    goToStage={goToStage}
                  />
                )}
                {stage.key === "training" && (
                  <TrainingStage key={session} runId={runId} run={run} onNewRun={newRun} goToStage={goToStage} />
                )}
                {stage.key === "results" && <ResultsStage key={session} runId={runId} run={run} goToStage={goToStage} />}
                {stage.key === "deploy" && <DeployStage key={session} runId={runId} run={run} goToStage={goToStage} />}
              </div>
            </section>
          ))}
        </div>
      </main>

      <Pager stages={STAGES.map((s, i) => ({ ...s, tone: tones[i] }))} active={active} onSelect={goTo} />
    </div>
  );
}

/** One status tone per stage, for the dots in the pager. */
function stageTones(runId: string | null, run: ReturnType<typeof useRun>): Tone[] {
  const { preview, previewError, status, statusError, result, resultError, deployment } = run;
  const data: Tone = !runId
    ? "idle"
    : previewError || statusError || isUploadFailed(status)
      ? "malfunction"
      : preview
        ? "done"
        : "running";
  const training: Tone = isRunActive(status)
    ? "running"
    : status?.status === "succeeded"
      ? "done"
      : isRunFailed(status)
        ? "malfunction"
        : "idle";
  const results: Tone =
    result ? "done" : resultError || isRunFailed(status) ? "malfunction" : status?.status === "succeeded" ? "running" : "idle";
  const deploy: Tone =
    deployment?.status === "succeeded"
      ? "done"
      : deployment?.status === "deploying"
        ? "running"
        : deployment?.status === "failed"
          ? "malfunction"
          : "idle";
  return [data, training, results, deploy];
}
