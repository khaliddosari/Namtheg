"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { StatusDot, type Tone } from "@/components/panel";

const TONE_WORD: Record<Tone, string> = {
  idle: "not started",
  running: "in progress",
  done: "done",
  malfunction: "needs attention",
};

/**
 * The stage switcher under the track: previous and next, and one button per
 * stage with its number, name and a status dot. It slides the track; nothing
 * reloads.
 */
export function Pager({
  stages,
  active,
  onSelect,
}: {
  stages: { key: string; label: string; tone: Tone }[];
  active: number;
  onSelect: (index: number) => void;
}) {
  const last = stages.length - 1;

  return (
    <nav
      aria-label="Stages"
      className="relative z-20 flex h-14 shrink-0 items-center justify-center gap-1.5 border-t border-primary/10 bg-background/65 px-2 backdrop-blur-xl backdrop-saturate-150 sm:gap-3"
    >
      <Tooltip>
        <TooltipTrigger
          render={
            <Button
              variant="ghost"
              size="icon"
              className="hidden size-10 sm:inline-flex lg:size-8"
              aria-label="Previous stage"
              disabled={active === 0}
              onClick={() => onSelect(active - 1)}
            />
          }
        >
          <ChevronLeft />
        </TooltipTrigger>
        <TooltipContent>Previous stage</TooltipContent>
      </Tooltip>

      <ol className="flex min-w-0 items-center gap-0.5 rounded-xl sm:gap-1 bg-muted p-1 ring-1 ring-(--glass-edge)">
        {stages.map((s, i) => (
          <li key={s.key}>
            <button
              type="button"
              onClick={() => onSelect(i)}
              aria-current={i === active ? "step" : undefined}
              aria-label={`${s.label}, ${TONE_WORD[s.tone]}`}
              className={cn(
                "flex h-10 items-center gap-1.5 rounded-lg px-2 text-sm font-medium text-muted-foreground transition-colors outline-none hover:bg-muted hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50 sm:gap-2 sm:px-3 lg:h-8",
                "aria-[current=step]:bg-primary/15 aria-[current=step]:text-primary aria-[current=step]:ring-1 aria-[current=step]:ring-primary/35 aria-[current=step]:ring-inset"
              )}
            >
              <span className="hidden font-mono text-xs tabular-nums sm:inline">{String(i + 1).padStart(2, "0")}</span>
              <span>{s.label}</span>
              <StatusDot tone={s.tone} />
            </button>
          </li>
        ))}
      </ol>

      <Tooltip>
        <TooltipTrigger
          render={
            <Button
              variant="ghost"
              size="icon"
              className="hidden size-10 sm:inline-flex lg:size-8"
              aria-label="Next stage"
              disabled={active === last}
              onClick={() => onSelect(active + 1)}
            />
          }
        >
          <ChevronRight />
        </TooltipTrigger>
        <TooltipContent>Next stage</TooltipContent>
      </Tooltip>

    </nav>
  );
}
