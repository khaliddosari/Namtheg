"use client";

import { ArrowRight, BrainCircuit, Cpu, MessageSquareQuote, Rocket, Upload } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Brand } from "@/components/brand";

// What a run goes through, in the order the stages show it.
const FLOW = [
  {
    icon: Upload,
    title: "Upload a table",
    text: "CSV, TSV, Excel, Parquet or JSON, up to 2 GB. Pick the column to predict.",
  },
  {
    icon: BrainCircuit,
    title: "The analyst investigates it",
    text: "An AI analyst runs code on your data in a sandbox: what each column means, leakage, gaps and the cleaning it needs.",
  },
  {
    icon: Cpu,
    title: "Models train on an NVIDIA H200",
    text: "Boosted trees, SVMs, nearest neighbours and linear models, cross-validated in parallel, then the best one is tuned.",
  },
  {
    icon: MessageSquareQuote,
    title: "Results with a grounded explanation",
    text: "The champion, its scores and the full leaderboard, explained only with numbers the run computed.",
  },
  {
    icon: Rocket,
    title: "Deploy for live predictions",
    text: "One click puts the model behind an HTTPS endpoint, with a test page and a QR code to share.",
  },
];

/**
 * The first slide: a poster that says what Namtheg does, from upload to a live
 * model, with one way in. It comes before the stages, so a bare link opens it.
 * Auto margins centre it, which, unlike centring the parent, keeps its top
 * reachable when it is taller than the slide.
 */
export function IntroStage({ onStart }: { onStart: () => void }) {
  return (
    <div className="flex flex-1 flex-col">
      <article
        aria-labelledby="intro-title"
        className="m-auto w-full max-w-6xl rounded-3xl bg-card shadow-(--glass-shadow) ring-1 ring-(--glass-edge) backdrop-blur-xl backdrop-saturate-150 lock-roomy:max-w-7xl"
      >
        <div className="grid gap-8 p-6 sm:p-10 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)] lg:gap-14 lock-short:gap-10 lock-short:p-7 lock-roomy:gap-20 lock-roomy:p-14">
          <div className="flex flex-col items-center justify-center gap-5 text-center lock-short:gap-4">
            {/* The wordmark over its Latin name. The size and the nudge were measured from the painted
                glyphs, so the two lines start and end together; both are in em, so they hold at any size. */}
            <p className="flex flex-col items-center text-5xl leading-none lock-short:text-4xl">
              <Brand />
              <span className="mt-[0.62em] -translate-x-[0.037em] text-[0.565em] font-bold text-muted-foreground">Namtheg</span>
            </p>
            <h2
              id="intro-title"
              className="text-3xl leading-tight font-bold tracking-tight text-balance sm:text-4xl lock:text-5xl lock-short:text-4xl lock-roomy:text-6xl"
            >
              From a raw table to a live model
            </h2>
            <p className="max-w-xl text-base text-pretty text-muted-foreground lock-roomy:text-lg">
              Namtheg studies your data like an analyst, trains every candidate model on an NVIDIA H200, explains the winner with
              numbers it actually computed, and serves it for live predictions.
            </p>
            <Button onClick={onStart} className="mt-1 h-11 px-5 text-base has-data-[icon=inline-end]:pe-4">
              Start with your data
              <ArrowRight data-icon="inline-end" />
            </Button>
          </div>

          <div className="flex flex-col justify-center">
            <h3 className="mb-4 text-xs font-semibold tracking-[0.18em] text-muted-foreground uppercase lock-short:mb-3 lock-roomy:mb-6">
              How it works
            </h3>
            <ol className="flex flex-col gap-4 lock-short:gap-3 lock-roomy:gap-6">
              {FLOW.map(({ icon: Icon, title, text }, i) => (
                <li key={title} className="relative flex gap-4">
                  {/* The line down to the next step: an element of its own rather than a pseudo-element, which would hide the text's background from contrast checks. */}
                  {i < FLOW.length - 1 && (
                    <span aria-hidden="true" className="absolute start-5 top-11 -bottom-4 w-px bg-primary/20 lock-short:-bottom-3 lock-roomy:-bottom-6" />
                  )}
                  <span className="flex size-10 shrink-0 items-center justify-center rounded-xl bg-primary/10 text-primary ring-1 ring-primary/25">
                    <Icon aria-hidden="true" className="size-5" />
                  </span>
                  <div className="min-w-0 pt-0.5">
                    <p className="font-bold lock-roomy:text-lg">
                      <span className="me-2 text-primary tabular-nums">{i + 1}</span>
                      {title}
                    </p>
                    <p className="text-sm text-muted-foreground lock-roomy:text-base">{text}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </div>
      </article>
    </div>
  );
}
