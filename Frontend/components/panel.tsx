import * as React from "react";
import { RotateCw, TriangleAlert } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * The four status tones: nothing happening, something in flight, finished well,
 * needs a human. The pager shows one per stage as a dot.
 */
export type Tone = "idle" | "running" | "done" | "malfunction";

const dotClass: Record<Tone, string> = {
  idle: "bg-slate-400",
  running: "bg-blue-400",
  done: "bg-green-400",
  malfunction: "bg-red-400",
};

export function StatusDot({ tone, className }: { tone: Tone; className?: string }) {
  return (
    <span aria-hidden="true" className={cn("relative flex size-1.5 shrink-0", className)}>
      {tone === "running" && (
        <span className={cn("absolute inline-flex size-full rounded-full opacity-60 motion-safe:animate-ping", dotClass[tone])} />
      )}
      <span className={cn("relative inline-flex size-1.5 rounded-full", dotClass[tone])} />
    </span>
  );
}

/**
 * The glass panel every card on the dashboard is built from. The header
 * is shrink-0 and the body min-h-0 flex-1, so on a locked page the body scrolls
 * inside its grid cell instead of stretching the panel.
 */
export function Panel({
  title,
  caption,
  actions,
  className,
  bodyClassName,
  children,
  ...props
}: Omit<React.ComponentProps<"section">, "title"> & {
  title: React.ReactNode;
  caption?: React.ReactNode;
  actions?: React.ReactNode;
  bodyClassName?: string;
}) {
  return (
    <section
      data-slot="panel"
      className={cn(
        "flex min-h-0 min-w-0 flex-col rounded-2xl bg-card text-card-foreground shadow-(--glass-shadow) ring-1 ring-(--glass-edge) backdrop-blur-xl backdrop-saturate-150 lock:overflow-hidden",
        className
      )}
      {...props}
    >
      <header className="shrink-0 border-b border-primary/15 px-4 pt-3 pb-2.5 lock-short:pt-2.5 lock-short:pb-2">
        <div className="flex min-w-0 items-center gap-2">
          <h2 className="min-w-0 truncate text-base font-bold tracking-tight">{title}</h2>
          {actions && <div className="ms-auto flex shrink-0 items-center gap-2">{actions}</div>}
        </div>
        {caption && <p className="mt-0.5 text-xs text-muted-foreground lock:truncate lock-short:hidden">{caption}</p>}
      </header>
      <div className={cn("flex min-h-0 flex-1 flex-col gap-3 p-4 lock:overflow-y-auto lock-short:p-3", bodyClassName)}>{children}</div>
    </section>
  );
}

/** What fills a panel before its stage has data: a mark, a heading, one sentence. */
export function PanelEmpty({
  icon,
  title,
  description,
  action,
  tone = "default",
  className,
}: {
  icon: React.ReactNode;
  title: string;
  description?: React.ReactNode;
  action?: React.ReactNode;
  tone?: "default" | "error";
  className?: string;
}) {
  return (
    <Empty className={cn("gap-3 p-4 lock-short:p-2", className)} role={tone === "error" ? "alert" : undefined}>
      <EmptyHeader className="gap-1.5">
        <EmptyMedia
          variant="icon"
          className={cn(
            "mb-1 size-10 rounded-xl ring-1 lock-short:hidden [&_svg:not([class*='size-'])]:size-5",
            tone === "error" ? "bg-red-500/15 text-red-300 ring-red-400/30" : "bg-primary/10 text-primary ring-primary/20"
          )}
        >
          {icon}
        </EmptyMedia>
        <EmptyTitle className="text-base">{title}</EmptyTitle>
        {description && <EmptyDescription className="text-xs">{description}</EmptyDescription>}
      </EmptyHeader>
      {action && <EmptyContent>{action}</EmptyContent>}
    </Empty>
  );
}

/** A load that failed: what went wrong and a way to try again. */
export function PanelError({
  title,
  message,
  onRetry,
  action,
}: {
  title: string;
  message?: string | null;
  onRetry?: () => void;
  action?: React.ReactNode;
}) {
  return (
    <PanelEmpty
      tone="error"
      icon={<TriangleAlert />}
      title={title}
      description={message ?? undefined}
      action={
        onRetry || action ? (
          <div className="flex flex-wrap items-center justify-center gap-2">
            {onRetry && (
              <Button variant="outline" onClick={onRetry} className="h-10 lg:h-8">
                <RotateCw data-icon="inline-start" />
                Try again
              </Button>
            )}
            {action}
          </div>
        ) : undefined
      }
    />
  );
}

/**
 * First load only: grey bars shaped roughly like the content, so a panel that
 * is fetching does not claim it has nothing to show.
 */
export function PanelSkeleton({ rows = 4, block = false, className }: { rows?: number; block?: boolean; className?: string }) {
  const widths = ["w-3/4", "w-1/2", "w-5/6", "w-2/3", "w-1/3", "w-4/5"];
  return (
    <div role="status" aria-label="Loading" className={cn("flex min-h-0 flex-1 flex-col gap-3", className)}>
      {block && <Skeleton className="min-h-24 flex-1 rounded-xl" />}
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className={cn("h-4", widths[i % widths.length])} />
      ))}
    </div>
  );
}

/** Label over value, one line each, full value in the title. */
export function Fact({
  label,
  value,
  title,
  className,
  valueClassName,
}: {
  label: string;
  value: React.ReactNode;
  title?: string;
  className?: string;
  valueClassName?: string;
}) {
  return (
    <div className={cn("min-w-0", className)}>
      <dt className="truncate text-xs text-muted-foreground">{label}</dt>
      <dd className={cn("mt-0.5 truncate text-sm leading-snug font-medium", valueClassName)} title={title}>
        {value}
      </dd>
    </div>
  );
}

/** A thin horizontal meter. Width only animates as a colour-free transition. */
export function Meter({ value, className, barClassName }: { value: number; className?: string; barClassName?: string }) {
  const pct = Math.max(0, Math.min(100, value));
  return (
    <div className={cn("h-1.5 w-full overflow-hidden rounded-full bg-muted", className)}>
      <div className={cn("h-full rounded-full bg-primary transition-[width] duration-500", barClassName)} style={{ width: `${pct}%` }} />
    </div>
  );
}
