"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import {
  downloadUrl,
  getResult,
  getStatus,
  plotUrl,
  type RunResult,
  type FeatureImportance,
  type ModelScore,
  type TuningTrial,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import { formatArabicBrand } from "@/components/brand";
import { Icon } from "@/components/icon";

// Stylesheets whose @font-face rules we inline (as base64) into the export
// snapshot so the report's text + Font Awesome icon glyphs rasterize correctly.
const FONT_EMBED_URLS = [
  "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap",
  "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css",
];
// Built once and reused across exports — the inlined webfont CSS is large and
// identical every time, so there's no reason to refetch it per click.
let fontEmbedCSSCache: string | null = null;

// Selection metrics the trainer can report (imbalanced targets use macro F1).
// percent: shown as a percentage; otherwise as a plain number (MASE and
// silhouette aren't proportions). lowerIsBetter flips ranking and wording.
const METRIC_LABELS: Record<string, { short: string; long: string; percent: boolean; lowerIsBetter?: boolean }> = {
  accuracy: { short: "Accuracy", long: "Cross-Validation Accuracy", percent: true },
  f1_macro: { short: "Macro F1", long: "Cross-Validation Macro F1", percent: true },
  r2: { short: "R²", long: "R² Regression Score", percent: true },
  silhouette: { short: "Silhouette", long: "Cluster Silhouette (-1 to 1)", percent: false },
  mase: { short: "MASE", long: "Backtest MASE (lower is better)", percent: false, lowerIsBetter: true },
};
const metricLabel = (metric: string) => METRIC_LABELS[metric] ?? { short: metric, long: metric, percent: false };
const formatScore = (value: number, metric: string, digits = 2) =>
  metricLabel(metric).percent ? `${(value * 100).toFixed(digits)}%` : value.toFixed(4);

/* ─── Metric card ── */
/* ─── Combined Accuracy + Feature Importance card ── */
function AccuracyAndFeaturesCard({
  features,
  accuracyPct,
  accuracyLabel,
  accuracyUnit,
  trendLabel,
  trendKind,
  trainScore,
  overfitGap,
}: {
  features: FeatureImportance[];
  accuracyPct: number;
  accuracyLabel: string;
  accuracyUnit: string;
  trendLabel: string;
  trendKind: "up" | "flat" | "down";
  trainScore?: number;
  overfitGap?: number;
}) {
  const top = features.slice(0, 5);
  const max = Math.max(...top.map((f) => f.importance));

  const trendColor =
    trendKind === "down" ? "text-error" : trendKind === "up" ? "text-success-green" : "text-on-surface-variant";
  const trendIcon = trendKind === "up" ? "trending_up" : trendKind === "down" ? "trending_down" : "horizontal_rule";

  return (
    <div className="glass p-4 sm:p-6 flex flex-col h-full relative overflow-hidden text-left">
      {/* Accuracy header */}
      <div className="flex justify-between items-start mb-4">
        <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-lg sm:rounded-xl shrink-0 bg-surface-green-tint text-success-green flex items-center justify-center">
          <Icon name="target" className="block" style={{ fontSize: "22px" }} />
        </div>
        <span className={cn("flex items-center text-sm font-bold gap-1 shrink-0", trendColor)}>
          <Icon name={trendIcon} style={{ fontSize: "15px" }} />
          {trendLabel}
        </span>
      </div>
      <h3 className="text-sm font-bold text-on-surface-variant uppercase tracking-wider mb-1">{accuracyLabel}</h3>
      <div className="text-3xl sm:text-4xl font-black text-success-green font-mono leading-none">
        {accuracyPct.toFixed(2)}
        <span className="text-base font-bold text-success-green/70 ml-1 font-sans">{accuracyUnit}</span>
      </div>
      <div className="mt-4 w-full bg-surface-variant h-2 rounded-full overflow-hidden">
        <motion.div
          className="bg-success-green h-full rounded-full"
          initial={{ width: 0 }}
          animate={{ width: `${accuracyPct}%` }}
          transition={{ duration: 0.9, ease: "easeOut" }}
        />
      </div>
      {trainScore !== undefined && (
        <div className="mt-3 flex flex-col sm:flex-row sm:justify-between gap-1">
          {overfitGap !== undefined && overfitGap > 0.12 && (
            <span className="text-warning-orange flex items-center gap-1.5 text-sm font-sans font-bold uppercase tracking-wider">
              <Icon name="warning" style={{ fontSize: "13px" }} />
              Overfit warning: {(overfitGap * 100).toFixed(0)}% Gap
            </span>
          )}
          <span className="text-sm font-bold text-on-surface-variant uppercase tracking-wider font-mono">
            Train score: {(trainScore * 100).toFixed(1)}%
          </span>
        </div>
      )}

      {/* Divider */}
      <div className="my-6 border-t border-outline-variant" />

      {/* Feature importance section */}
      <div className="flex justify-between items-center mb-5">
        <h3 className="text-lg sm:text-headline-md font-bold text-on-background">Feature Importance</h3>
        <Icon name="analytics" className="text-outline" style={{ fontSize: "18px" }} />
      </div>

      <div className="flex-1 flex flex-col gap-4 relative">
        {top.map((f, i) => {
          const pct = max > 0 ? (f.importance / max) * 100 : 0;
          const alpha = 0.15 + (pct / 100) * 0.85;
          return (
            <div key={f.feature} className="relative flex-1 flex flex-col justify-center min-h-0">
              <div className="flex justify-between text-sm font-bold mb-2">
                <span className="text-on-background truncate mr-2">{f.feature}</span>
                <span className="text-on-surface-variant font-mono">{f.importance.toFixed(4)}</span>
              </div>
              <div className="w-full bg-surface-variant h-3 sm:h-4 rounded-full overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  style={{ background: `rgba(79, 195, 247, ${alpha})` }}
                  initial={{ width: 0 }}
                  animate={{ width: `${pct}%` }}
                  transition={{ duration: 0.65, delay: i * 0.06, ease: "easeOut" }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ─── Full Plot Lightbox Zoom ── */
function ResultPlotCard({ runId, problemType }: { runId: string; problemType?: string }) {
  const [loaded, setLoaded] = useState(false);
  const [hasError, setHasError] = useState(false);
  const [lightboxOpen, setLightboxOpen] = useState(false);
  const imgRef = useRef<HTMLImageElement>(null);
  const isClassification = problemType === "classification";
  const url = plotUrl(runId);

  useEffect(() => {
    if (imgRef.current && imgRef.current.complete && imgRef.current.naturalWidth > 0) {
      setLoaded(true);
    }
  }, [url]);

  return (
    <>
      <div className="glass p-4 sm:p-6 flex flex-col h-full relative overflow-hidden text-left">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-headline-md font-bold text-on-background">
            {isClassification ? "Confusion Matrix" : "Predicted vs Actual"}
          </h3>
          <button
            onClick={() => setLightboxOpen(true)}
            className="text-primary hover:bg-surface-purple-tint p-1.5 rounded-lg transition-all flex items-center gap-1 text-xs font-bold"
          >
            <Icon name="zoom_in" style={{ fontSize: "15px" }} />
            Zoom
          </button>
        </div>

        <div
          onClick={() => setLightboxOpen(true)}
          className="flex-1 relative rounded-lg overflow-hidden bg-transparent min-h-[220px] cursor-pointer group flex items-center justify-center"
        >
          {!loaded && !hasError && <div className="absolute inset-0 shimmer" />}

          {/* Hover overlay */}
          <div className="absolute inset-0 bg-black/30 opacity-0 group-hover:opacity-100 flex items-center justify-center transition-all duration-300 z-10">
            <Icon name="zoom_in" className="text-primary bg-surface-bright p-3 rounded-full shadow-md" style={{ fontSize: "28px" }} />
          </div>

          <img
            ref={imgRef}
            src={url}
            alt="Model result plot"
            className={cn(
              "w-full h-full object-contain transition-all duration-300 group-hover:scale-103",
              loaded ? "opacity-100" : "opacity-90"
            )}
            onLoad={() => setLoaded(true)}
            onError={() => setHasError(true)}
          />
        </div>
        <div className="text-xs font-medium text-on-surface-variant mt-3 uppercase tracking-wider font-mono space-y-1">
          {isClassification ? (
            <>
              <p>Diagonal = correct predictions.</p>
              <p>Bright cyan cells indicate accurate validation.</p>
            </>
          ) : (
            <p>Closer plot clusters along correlation slope indicate tighter residual coefficients.</p>
          )}
        </div>
      </div>

      {/* Lightbox Modal */}
      <AnimatePresence>
        {lightboxOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setLightboxOpen(false)}
            className="fixed inset-0 bg-black/80 backdrop-blur-md z-50 flex items-center justify-center p-6 cursor-zoom-out"
          >
            <motion.div
              initial={{ scale: 0.95 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.95 }}
              onClick={(e) => e.stopPropagation()}
              className="glass-strong rounded-2xl p-6 shadow-2xl max-w-2xl w-full relative"
            >
              <button
                onClick={() => setLightboxOpen(false)}
                aria-label="Close"
                className="absolute top-4 right-4 text-outline hover:text-primary hover:bg-surface-container p-1 rounded-full transition-colors cursor-pointer"
              >
                <Icon name="close" className="block" />
              </button>

              <h3 className="text-headline-md font-bold text-on-surface mb-4">
                {isClassification ? "Confusion Matrix (High Res)" : "Predicted vs Actual Residuals"}
              </h3>

              <div className="bg-surface-container rounded-xl p-2 border border-outline-variant flex items-center justify-center max-h-[500px]">
                <img src={url} alt="Large plot" crossOrigin="anonymous" className="max-h-[460px] object-contain rounded-lg" />
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

/* ─── Hyperparameter Tuning Before/After Card ── */
function TuningResultsCard({
  trials,
  baseline,
  optimized,
  metric,
  modelName,
}: {
  trials: TuningTrial[];
  baseline: number;
  optimized: number;
  metric: string;
  modelName: string;
}) {
  const improved = optimized > baseline + 0.0001;
  const deltaPct = ((optimized - baseline) * 100).toFixed(2);
  const sign = improved ? "+" : "";
  const tuningTrials = trials.filter((t) => t.trial > 0);

  return (
    <div className="glass overflow-hidden text-left">
      {/* Header */}
      <div className="p-4 sm:p-5 border-b border-outline-variant flex flex-wrap justify-between items-center gap-3 bg-surface-container-low">
        <div>
          <h3 className="text-xl sm:text-[28px] leading-tight font-bold tracking-tight text-on-background">Hyperparameter Tuning</h3>
          <p className="text-xs text-on-surface-variant mt-1.5 font-mono">
            Bayesian search (Optuna) · {tuningTrials.length} trial{tuningTrials.length !== 1 ? "s" : ""} on <strong className="text-primary">{modelName}</strong>
          </p>
        </div>
        <span className={cn(
          "hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold border shrink-0",
          improved
            ? "bg-surface-green-tint text-success-green border-success-green/20"
            : "bg-surface-container text-outline border-outline-variant"
        )}>
          <Icon name={improved ? "trending_up" : "horizontal_rule"} style={{ fontSize: "13px" }} />
          {improved ? `Improved ${sign}${deltaPct}%` : "No Improvement Found"}
        </span>
      </div>

      {/* Before vs After panels */}
      <div className="grid grid-cols-1 sm:grid-cols-2 sm:divide-x divide-y sm:divide-y-0 divide-outline-variant border-b border-outline-variant">
        {/* Before */}
        <div className="p-4 sm:p-6 flex flex-col gap-1">
          <span className="text-[10px] font-black uppercase tracking-widest text-outline font-mono mb-1">
            ◎ Before Tuning
          </span>
          <span className="text-3xl sm:text-4xl font-black font-mono text-on-surface-variant">
            {(baseline * 100).toFixed(2)}
            <span className="text-base font-bold text-outline ml-1">%</span>
          </span>
          <span className="text-[11px] text-on-surface-variant font-mono mt-0.5">
            Baseline {metricLabel(metric).short} · default params
          </span>
          <div className="mt-3 h-1.5 bg-surface-variant rounded-full overflow-hidden">
            <div className="h-full bg-outline/30 rounded-full" style={{ width: `${Math.min(baseline * 100, 100)}%` }} />
          </div>
        </div>
        {/* After */}
        <div className="p-4 sm:p-6 flex flex-col gap-1 relative">
          <span className="text-[10px] font-black uppercase tracking-widest font-mono mb-1 text-success-green">
            ✦ After Tuning
          </span>
          <span className="text-3xl sm:text-4xl font-black font-mono text-success-green">
            {(optimized * 100).toFixed(2)}
            <span className="text-base font-bold text-success-green/70 ml-1">%</span>
          </span>
          <span className="text-[11px] text-on-surface-variant font-mono mt-0.5">
            Optimized {metricLabel(metric).short} · best params
          </span>
          <div className="mt-3 h-1.5 bg-surface-variant rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-success-green rounded-full"
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(optimized * 100, 100)}%` }}
              transition={{ duration: 0.9, ease: "easeOut" }}
            />
          </div>
        </div>
      </div>

      {/* Trial history table */}
      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse font-sans text-xs sm:text-sm">
          <thead>
            <tr className="bg-surface-container-low text-label-md text-on-surface-variant font-bold border-b border-outline-variant">
              <th className="p-2 sm:p-3 text-[10px] sm:text-xs uppercase tracking-wider w-16 sm:w-20 whitespace-nowrap text-center">Trial</th>
              <th className="p-2 sm:p-3 text-[10px] sm:text-xs uppercase tracking-wider whitespace-nowrap hidden sm:table-cell">Parameters Tested</th>
              <th className="p-2 sm:p-3 text-[10px] sm:text-xs uppercase tracking-wider text-center w-20 sm:w-28 whitespace-nowrap">Score</th>
              <th className="p-2 sm:p-3 text-[10px] sm:text-xs uppercase tracking-wider whitespace-nowrap">Outcome</th>
            </tr>
          </thead>
          <tbody>
            {trials.map((t, idx) => {
              const isBaseline = t.trial === 0;
              const isBest = !isBaseline && t.score !== null && Math.abs(t.score - optimized) < 0.00005 && improved;

              // Parse the trial parameters into a list of {key, value} chips so
              // the user can see exactly which hyperparameters the optimizer
              // changed for this trial. Falls back to raw string on parse error.
              let paramEntries: Array<[string, string]> = [];
              let paramFallback = "";
              if (!isBaseline) {
                try {
                  const obj = JSON.parse(t.parameters) as Record<string, unknown>;
                  paramEntries = Object.entries(obj).map(([k, v]) => [k, String(v)]);
                } catch {
                  paramFallback = t.parameters;
                }
              }

              return (
                <tr
                  key={idx}
                  className={cn(
                    "border-b border-outline-variant/40 last:border-0 transition-colors",
                    isBest && "bg-surface-green-tint",
                    !isBest && "hover:bg-surface-container-low"
                  )}
                >
                  {/* Trial # */}
                  <td className="p-2 sm:p-3 align-middle text-center">
                    {isBaseline ? (
                      <span className="font-mono font-bold text-primary">#0</span>
                    ) : isBest ? (
                      <span className="font-mono font-bold text-success-green">#{t.trial}</span>
                    ) : (
                      <span className="font-mono font-bold text-on-surface-variant">#{t.trial}</span>
                    )}
                  </td>
                  {/* Params — render each changed hyperparameter as a chip */}
                  <td className="p-2 sm:p-3 align-middle hidden sm:table-cell">
                    {isBaseline ? (
                      <span className="font-mono text-primary italic">
                        Model defaults
                      </span>
                    ) : paramEntries.length > 0 ? (
                      <div className="flex flex-nowrap gap-1.5 whitespace-nowrap">
                        {paramEntries.map(([k, v]) => (
                          <span
                            key={k}
                            className={cn(
                              "inline-flex items-baseline gap-1 px-2.5 py-1 rounded-md font-mono text-xs whitespace-nowrap",
                              isBest
                                ? "border bg-surface-green-tint border-success-green/20"
                                : "btn-glass"
                            )}
                          >
                            <span className={cn("font-bold", isBest ? "text-success-green" : "text-primary")}>{k}</span>
                            <span className="text-outline">=</span>
                            <span className="font-semibold text-on-background">{v}</span>
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="font-mono text-on-surface-variant break-all">
                        {paramFallback}
                      </span>
                    )}
                  </td>
                  {/* Score */}
                  <td className="p-2 sm:p-3 text-center font-mono font-bold align-middle">
                    <span className={cn(
                      isBest ? "text-success-green" :
                      isBaseline ? "text-primary" : "text-on-surface"
                    )}>
                      {t.score === null ? "—" : `${(t.score * 100).toFixed(2)}%`}
                    </span>
                  </td>
                  {/* Outcome — keep the verdict+scores in full, trim the LLM reason
                     so rows stay compact. Full text is shown on hover. */}
                  <td className="p-2 sm:p-3 align-middle">
                    {(() => {
                      const display = (() => {
                        if (isBaseline) return "Baseline: champion from initial leaderboard sweep";
                        const reasonIdx = t.result.indexOf("Reason:");
                        if (reasonIdx === -1) {
                          // No reason segment — cap whole string at 110 chars.
                          return t.result.length > 110 ? t.result.slice(0, 107) + "…" : t.result;
                        }
                        const verdict = t.result.slice(0, reasonIdx).trim();
                        const reason = t.result.slice(reasonIdx + "Reason:".length).trim();
                        const shortReason = reason.length > 90 ? reason.slice(0, 87) + "…" : reason;
                        return `${verdict} Reason: ${shortReason}`;
                      })();
                      return (
                        <span
                          title={t.result}
                          className={cn(
                            "font-medium leading-relaxed whitespace-normal break-words block",
                            isBaseline && "text-primary font-semibold",
                            t.result.toLowerCase().includes("new champion") && "text-success-green font-bold",
                            t.result.toLowerCase().includes("error") && "text-error",
                            !isBaseline && !t.result.toLowerCase().includes("new champion") && !t.result.toLowerCase().includes("error") && "text-on-surface-variant"
                          )}
                        >
                          {formatArabicBrand(display)}
                        </span>
                      );
                    })()}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ─── Result page ── */
export default function ResultPage() {
  const { runId } = useParams<{ runId: string }>();
  const router = useRouter();
  const [result, setResult] = useState<RunResult | null>(null);
  const [loading, setLoading] = useState(true);


  // PNG export
  const exportRef = useRef<HTMLDivElement>(null);
  const [exporting, setExporting] = useState(false);

  // Captures the report region into a fully-styled PNG dataURL. Shared by both
  // the PNG and PDF export handlers so they always render identically.
  const captureSnapshot = async (): Promise<{
    dataUrl: string;
    width: number;
    height: number;
  } | null> => {
    if (!exportRef.current) return null;

    const fetchAsDataUrl = async (url: string): Promise<string | null> => {
      try {
        const res = await fetch(url, { mode: "cors" });
        if (!res.ok) return null;
        const blob = await res.blob();
        return await new Promise<string>((resolve, reject) => {
          const fr = new FileReader();
          fr.onloadend = () => resolve(fr.result as string);
          fr.onerror = reject;
          fr.readAsDataURL(blob);
        });
      } catch {
        return null;
      }
    };

    // Fetches the Google Fonts CSS and inlines every woff2 URL inside it as a
    // base64 data URI, so the resulting stylesheet is fully self-contained and
    // works inside the SVG <foreignObject> that html-to-image rasterizes.
    const fetchFontEmbedCSS = async (cssUrl: string): Promise<string> => {
      try {
        const res = await fetch(cssUrl, { mode: "cors" });
        if (!res.ok) return "";
        let css = await res.text();
        // Capture every url(...) reference (quotes optional). Font Awesome's
        // all.min.css points at its webfonts with *relative* paths, so we resolve
        // each ref against the stylesheet URL before inlining it as base64.
        const rawRefs = Array.from(
          new Set(
            Array.from(css.matchAll(/url\(\s*['"]?([^'")]+)['"]?\s*\)/g)).map((m) => m[1])
          )
        ).filter((u) => u && !u.startsWith("data:"));
        const replacements = await Promise.all(
          rawRefs.map(async (raw) => {
            let abs = raw;
            try {
              abs = new URL(raw, cssUrl).href;
            } catch {
              /* keep raw */
            }
            return { raw, data: await fetchAsDataUrl(abs) };
          })
        );
        for (const { raw, data } of replacements) {
          if (data) css = css.split(raw).join(data);
        }
        return css;
      } catch {
        return "";
      }
    };

    const node = exportRef.current;
    const CAPTURE_WIDTH = 1280;

    // Build a sandbox container OUTSIDE the page's flex layout so we can render
    // a 1280px-wide copy of the report regardless of the user's viewport width.
    const sandbox = document.createElement("div");
    sandbox.style.position = "fixed";
    sandbox.style.left = "-100000px";
    sandbox.style.top = "0";
    sandbox.style.width = `${CAPTURE_WIDTH}px`;
    sandbox.style.zIndex = "-1";
    sandbox.style.pointerEvents = "none";
    sandbox.style.background = "#0a0a0f";

    const clone = node.cloneNode(true) as HTMLElement;
    clone.style.width = `${CAPTURE_WIDTH}px`;
    clone.style.maxWidth = `${CAPTURE_WIDTH}px`;
    clone.style.flex = "none";
    // A little bottom padding so the last row of the leaderboard isn't flush
    // against the canvas edge.
    clone.style.paddingBottom = "32px";
    clone
      .querySelectorAll("[data-export-ignore]")
      .forEach((el) => el.remove());

    // Strip scroll bars from the snapshot. The leaderboard and trial tables use
    // `overflow-x-auto`; at 1280px-wide their content fits, but some browsers
    // still reserve a scroll-bar gutter that appears as a grey strip in the
    // capture. Forcing `overflow: visible` on the clone subtree removes those
    // gutters while also hiding any vertical scrollbar artifacts.
    const styleReset = document.createElement("style");
    styleReset.textContent = `
      .__export-clone, .__export-clone * {
        overflow: visible !important;
        scrollbar-width: none !important;
      }
      .__export-clone ::-webkit-scrollbar,
      .__export-clone::-webkit-scrollbar { display: none !important; }
    `;
    clone.classList.add("__export-clone");
    sandbox.appendChild(styleReset);

    // Force the desktop layout regardless of the user's actual viewport. The
    // clone is rendered at 1280px, but Tailwind's `sm:`/`md:`/`lg:`/`xl:`
    // variants are media-query-gated on viewport width — on a phone they never
    // activate, so the snapshot keeps the cramped mobile layout. We harvest
    // every `min-width` media rule from the loaded stylesheets, re-emit each
    // inner rule with its selectors scoped to `.__export-clone`, and inject
    // the result so those desktop rules win the cascade only inside the clone.
    const desktopCSS = (() => {
      const out: string[] = [];
      const scopeSelector = (selector: string): string => {
        const t = selector.trim();
        if (!t) return "";
        // Don't try to scope root-level selectors — they wouldn't apply inside
        // the clone anyway and produce invalid CSS when prefixed.
        if (/^(html|body|:root)\b/.test(t)) return "";
        return `.__export-clone ${t}, .__export-clone${t}`;
      };
      const emitStyleRule = (rule: CSSStyleRule) => {
        const sel = rule.selectorText
          .split(",")
          .map(scopeSelector)
          .filter(Boolean)
          .join(", ");
        if (sel && rule.style.cssText) {
          out.push(`${sel} { ${rule.style.cssText} }`);
        }
      };
      const visit = (rules: CSSRuleList) => {
        for (const rule of Array.from(rules)) {
          if (rule instanceof CSSMediaRule) {
            const mt = rule.media.mediaText;
            const minMatch = mt.match(/min-width:\s*(\d+(?:\.\d+)?)px/);
            const hasMax = /max-width/.test(mt);
            if (
              minMatch &&
              !hasMax &&
              parseFloat(minMatch[1]) <= CAPTURE_WIDTH
            ) {
              for (const inner of Array.from(rule.cssRules)) {
                if (inner instanceof CSSStyleRule) emitStyleRule(inner);
              }
            }
          } else if (rule instanceof CSSSupportsRule) {
            visit(rule.cssRules);
          } else if (rule.constructor.name === "CSSLayerBlockRule") {
            // Tailwind v4 wraps its rules in `@layer utilities { ... }`; recurse
            // through the layer so its inner @media rules are reachable.
            visit((rule as unknown as { cssRules: CSSRuleList }).cssRules);
          }
        }
      };
      for (const sheet of Array.from(document.styleSheets)) {
        try {
          visit(sheet.cssRules);
        } catch {
          // Cross-origin stylesheet (e.g. Google Fonts) — skip silently.
        }
      }
      return out.join("\n");
    })();
    const desktopStyle = document.createElement("style");
    desktopStyle.textContent = desktopCSS;
    sandbox.appendChild(desktopStyle);

    sandbox.appendChild(clone);
    document.body.appendChild(sandbox);

    try {
      // Let the cloned subtree paint and any inline-styled animations settle.
      await new Promise((r) => requestAnimationFrame(() => r(null)));
      await new Promise((r) => requestAnimationFrame(() => r(null)));
      // Wait for every <img> in the clone to be ready (the backend plot is CORS).
      const imgs = Array.from(clone.querySelectorAll("img"));
      await Promise.all(
        imgs.map((img) =>
          img.complete && img.naturalWidth > 0
            ? Promise.resolve()
            : new Promise<void>((resolve) => {
                img.addEventListener("load", () => resolve(), { once: true });
                img.addEventListener("error", () => resolve(), { once: true });
              })
        )
      );

      // Make sure the live page's webfaces (Inter, IBM Plex, JetBrains Mono, and
      // the Font Awesome icon faces) are fully loaded before snapshotting — the
      // clone inherits them, and FA renders icons via ::before glyphs that won't
      // rasterize until the face is ready.
      if (document.fonts?.ready) {
        try {
          await document.fonts.ready;
        } catch {
          /* font loading API unavailable — proceed */
        }
      }

      // Inline @font-face rules with base64 woff2 so the report's text and the
      // Font Awesome icon glyphs render in the snapshot. Built once, then reused.
      if (fontEmbedCSSCache === null) {
        fontEmbedCSSCache = (
          await Promise.all(FONT_EMBED_URLS.map(fetchFontEmbedCSS))
        ).join("\n");
      }
      const fontEmbedCSS = fontEmbedCSSCache;

      // Wait one more frame after font load to make sure final layout is stable,
      // then take the maximum of every height measurement we can get to defend
      // against scrollHeight under-reporting when the clone is in a fixed sandbox.
      await new Promise((r) => requestAnimationFrame(() => r(null)));
      const rect = clone.getBoundingClientRect();
      const height = Math.ceil(
        Math.max(clone.scrollHeight, clone.offsetHeight, rect.height)
      );

      const { toPng } = await import("html-to-image");
      const dataUrl = await toPng(clone, {
        backgroundColor: "#0a0a0f",
        pixelRatio: 2,
        cacheBust: true,
        width: CAPTURE_WIDTH,
        height,
        fontEmbedCSS,
      });

      return { dataUrl, width: CAPTURE_WIDTH, height };
    } finally {
      document.body.removeChild(sandbox);
    }
  };

  const handleExportPng = async () => {
    if (exporting) return;
    setExporting(true);
    try {
      const snap = await captureSnapshot();
      if (!snap) return;
      const link = document.createElement("a");
      link.href = snap.dataUrl;
      link.download = `namtheg-${runId}-report.png`;
      link.click();
    } catch (e) {
      console.error("PNG export failed", e);
    } finally {
      setExporting(false);
    }
  };

  const handleExportPdf = async () => {
    if (exporting) return;
    setExporting(true);
    try {
      const snap = await captureSnapshot();
      if (!snap) return;

      // Convert the PNG dataURL to a JPEG at 92% quality. PNG is lossless and
      // would balloon the PDF past 40MB for a tall report; JPEG at 0.92 is
      // visually indistinguishable and an order of magnitude smaller.
      const jpegDataUrl = await new Promise<string>((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement("canvas");
          canvas.width = img.naturalWidth;
          canvas.height = img.naturalHeight;
          const ctx = canvas.getContext("2d");
          if (!ctx) return reject(new Error("2d context unavailable"));
          ctx.fillStyle = "#0a0a0f";
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.drawImage(img, 0, 0);
          resolve(canvas.toDataURL("image/jpeg", 0.92));
        };
        img.onerror = () => reject(new Error("snapshot failed to decode"));
        img.src = snap.dataUrl;
      });

      const { jsPDF } = await import("jspdf");
      // Build the PDF in points (1px = 0.75pt at 96dpi). Using 'pt' with an
      // explicit pixel→point conversion gives jsPDF a single accurate page
      // size that exactly matches the snapshot — no auto-pagination, no
      // letterboxing, just one tall page.
      const PX_TO_PT = 0.75;
      const widthPt = snap.width * PX_TO_PT;
      const heightPt = snap.height * PX_TO_PT;
      const pdf = new jsPDF({
        orientation: widthPt > heightPt ? "landscape" : "portrait",
        unit: "pt",
        format: [widthPt, heightPt],
        compress: true,
      });
      pdf.addImage(jpegDataUrl, "JPEG", 0, 0, widthPt, heightPt);
      pdf.save(`namtheg-${runId}-report.pdf`);
    } catch (e) {
      console.error("PDF export failed", e);
    } finally {
      setExporting(false);
    }
  };



  useEffect(() => {
    let cancelled = false;
    let interval: ReturnType<typeof setInterval> | null = null;

    const tryFetch = async () => {
      try {
        const r = await getResult(runId);
        if (cancelled) return;
        setResult(r);
        setLoading(false);
        if (interval) clearInterval(interval);
      } catch {
        // Result not ready yet — fall back to polling status. The run might still be
        // queued/running, in which case we wait and retry.
        try {
          const s = await getStatus(runId);
          if (cancelled) return;
          if (s.status === "failed") {
            setResult({ run_id: runId, status: "failed", error: s.error ?? "Run failed" });
            setLoading(false);
            if (interval) clearInterval(interval);
          }
          // otherwise keep polling — loading state stays on
        } catch {
          /* transient, keep polling */
        }
      }
    };

    tryFetch();
    interval = setInterval(tryFetch, 2000);
    return () => {
      cancelled = true;
      if (interval) clearInterval(interval);
    };
  }, [runId]);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center select-none">
        <div className="flex flex-col items-center gap-3">
          <Icon name="sync" className="text-primary animate-spin" style={{ fontSize: "36px" }} />
          <p className="text-sm font-semibold text-on-surface-variant font-mono">Loading model analysis...</p>
        </div>
      </div>
    );
  }

  if (!result || result.status === "failed") {
    return (
      <div className="flex-1 flex items-center justify-center flex-col gap-4 select-none">
        <Icon name="error_outline" className="text-error animate-pulse" style={{ fontSize: "52px" }} />
        <p className="text-sm font-semibold text-on-surface-variant font-mono">{result?.error ?? "No model metrics generated."}</p>
        <button
          onClick={() => router.push("/")}
          className="text-xs font-bold text-primary border border-outline-variant px-5 py-2.5 rounded-lg hover:bg-surface-purple-tint transition-all"
        >
          ← Start New Ingestion
        </button>
      </div>
    );
  }

  const extra = result.extra ?? {};
  const features: FeatureImportance[] = extra.top_features ?? [];
  const models: ModelScore[] = extra.all_models ?? [];
  const trainScore: number | undefined = extra.train_score ?? extra.train_accuracy ?? extra.train_r2;
  const overfitGap: number | undefined = extra.overfit_gap;
  const modelName = result.model_name ?? "Best Model";
  const score = result.accuracy_score ?? 0;
  const metric = result.score_metric ?? "score";
  const pct = Math.max(0, Math.min(100, score * 100));



  // Sort model scores for standard comparison
  const lowerIsBetter = result.higher_is_better === false || !!metricLabel(metric).lowerIsBetter;
  const sortedModels = [...models].sort((a, b) => (lowerIsBetter ? a.cv_mean - b.cv_mean : b.cv_mean - a.cv_mean));
  const winnerModel = models.find(m => m.name === modelName);

  return (
    <div className="flex-1 overflow-y-auto px-3 py-4 sm:p-gutter md:pb-24 relative select-none">

      <div ref={exportRef} className="max-w-[1280px] mx-auto w-full space-y-5 sm:space-y-8">

        {/* Page Header */}
        <motion.div
          initial={{ opacity: 0, y: -8 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-col md:flex-row md:items-end justify-between gap-4 text-left"
        >
          <div>
            <h1 className="text-headline-lg-mobile md:text-headline-lg font-bold text-on-background mb-2">Performance Analytics</h1>
            <p className="text-body-md text-on-surface-variant">
              Model performance reports for predicting <strong className="text-primary font-bold">{result.target}</strong> utilizing <strong className="text-primary font-bold">{modelName}</strong>.
            </p>
          </div>
          
          <div className="grid grid-cols-3 sm:flex sm:flex-wrap items-center gap-2 sm:gap-3 shrink-0 w-full sm:w-auto">
            <button
              onClick={() => router.push("/")}
              className="btn-glass text-xs px-3 sm:px-4 py-2 rounded-lg flex items-center justify-center sm:justify-start gap-1.5 w-full sm:w-auto"
            >
              <Icon name="refresh" style={{ fontSize: "15px" }} />
              New Run
            </button>
            <button
              onClick={handleExportPng}
              disabled={exporting}
              data-export-ignore
              className="btn-glass flex text-xs px-3 sm:px-4 py-2 rounded-lg items-center justify-center sm:justify-start gap-1.5 w-full sm:w-auto disabled:opacity-60 disabled:cursor-not-allowed"
            >
              <Icon
                name={exporting ? "sync" : "image"}
                className={cn(exporting && "animate-spin")}
                style={{ fontSize: "15px" }}
              />
              <span className="hidden sm:inline">{exporting ? "Exporting..." : "Export PNG"}</span>
              <span className="sm:hidden">{exporting ? "..." : "Export"}</span>
            </button>
            <button
              onClick={handleExportPdf}
              disabled={exporting}
              data-export-ignore
              className="btn-primary flex text-xs px-3 sm:px-4 py-2 rounded-lg items-center justify-center sm:justify-start gap-1.5 w-full sm:w-auto disabled:opacity-60 disabled:cursor-not-allowed"
            >
              <Icon
                name={exporting ? "sync" : "picture_as_pdf"}
                className={cn(exporting && "animate-spin")}
                style={{ fontSize: "15px" }}
              />
              <span className="hidden sm:inline">{exporting ? "Exporting..." : "Export PDF"}</span>
              <span className="sm:hidden">{exporting ? "..." : "PDF"}</span>
            </button>
          </div>
        </motion.div>

        {/* Champion Model Badge Certificate Card */}
        <motion.section
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass p-4 sm:p-6 flex flex-col md:flex-row md:items-stretch justify-between gap-4 sm:gap-6 text-left"
        >
          <div className="flex flex-col gap-3 flex-1 min-w-0">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-lg sm:rounded-xl bg-surface-purple-tint flex items-center justify-center shrink-0">
                <Icon name="workspace_premium" className="text-primary" style={{ fontSize: "22px" }} />
              </div>
              <h2 className="text-xl sm:text-2xl font-black text-on-surface truncate">{modelName}</h2>
            </div>
            {result.justification && (
              <p className="text-sm text-on-surface-variant leading-relaxed font-medium">
                {formatArabicBrand(result.justification)}
              </p>
            )}
          </div>

        </motion.section>

        {/* Inference CTA — deep link to the dedicated inference page */}
        <motion.button
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.07 }}
          onClick={() => router.push(`/${runId}/inference`)}
          className="glass glass-hover w-full p-4 sm:p-6 text-left group cursor-pointer flex items-center gap-3 sm:gap-4"
        >
          <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-lg sm:rounded-xl bg-surface-purple-tint flex items-center justify-center shrink-0 group-hover:scale-105 transition-transform">
            <Icon name="rocket_launch" className="text-primary" style={{ fontSize: "22px" }} />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="text-base sm:text-headline-md font-bold text-on-background">Deploy & try this model</h3>
            <p className="text-[11px] sm:text-xs font-medium text-on-surface-variant mt-0.5">
              Push the winning model to Modal as an API and run live predictions from the Inference page.
            </p>
          </div>
          <span
            className="btn-primary relative overflow-hidden shrink-0 px-3 sm:px-5 py-2 sm:py-2.5 rounded-lg text-sm group/try"
          >
            {/* Shimmer overlay — same treatment as the "Create New Model" button */}
            <span className="absolute inset-0 w-full h-full bg-gradient-to-r from-transparent via-white/25 to-transparent -translate-x-full group-hover:animate-shimmer group-hover/try:animate-shimmer" style={{ backgroundSize: "200% 100%" }} />
            <span className="z-10 font-semibold whitespace-nowrap hidden sm:inline">Try Now</span>
            <Icon
              name="arrow_forward"
              className="z-10 group-hover:translate-x-0.5 transition-transform"
              style={{ fontSize: "16px" }}
            />
          </span>
        </motion.button>

        {/* Downloads: the cleaned dataset and the winning model package */}
        {result.downloads && result.downloads.length > 0 && (
          <motion.section
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            data-export-ignore
            className="glass p-4 sm:p-6 flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-4 text-left"
          >
            <div className="w-10 h-10 sm:w-12 sm:h-12 rounded-lg sm:rounded-xl bg-surface-purple-tint flex items-center justify-center shrink-0">
              <Icon name="download" className="text-primary" style={{ fontSize: "22px" }} />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-base sm:text-headline-md font-bold text-on-background">Downloads</h3>
              <p className="text-[11px] sm:text-xs font-medium text-on-surface-variant mt-0.5">
                The cleaned data the models trained on, and the winning model with its weights and a predict script.
              </p>
            </div>
            <div className="grid grid-cols-2 sm:flex gap-2 shrink-0">
              {result.downloads.includes("cleaned_csv") && (
                <a
                  href={downloadUrl(runId, "cleaned_csv")}
                  className="btn-glass text-xs px-3 sm:px-4 py-2 rounded-lg flex items-center justify-center gap-1.5"
                >
                  <Icon name="table_view" style={{ fontSize: "15px" }} />
                  Cleaned CSV
                </a>
              )}
              {result.downloads.includes("forecast") && (
                <a
                  href={downloadUrl(runId, "forecast")}
                  className="btn-glass text-xs px-3 sm:px-4 py-2 rounded-lg flex items-center justify-center gap-1.5"
                >
                  <Icon name="table_view" style={{ fontSize: "15px" }} />
                  Forecast CSV
                </a>
              )}
              {result.downloads.includes("model") && (
                <a
                  href={downloadUrl(runId, "model")}
                  className="btn-primary text-xs px-3 sm:px-4 py-2 rounded-lg flex items-center justify-center gap-1.5"
                >
                  <Icon name="deployed_code" style={{ fontSize: "15px" }} />
                  Model + weights
                </a>
              )}
            </div>
          </motion.section>
        )}

        {/* Accuracy + Feature Importance (left) and Plot Graphics (right) */}
        <motion.section
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.12 }}
          className="grid grid-cols-1 lg:grid-cols-2 gap-6"
        >
          {features.length > 0 ? (
            <AccuracyAndFeaturesCard
              features={features}
              accuracyPct={pct}
              accuracyLabel={metricLabel(metric).long}
              accuracyUnit="%"
              trendLabel={pct >= 85 ? "Excellent Match" : "Moderate Accuracy"}
              trendKind={pct >= 85 ? "up" : pct >= 65 ? "flat" : "down"}
              trainScore={trainScore}
              overfitGap={overfitGap}
            />
          ) : (
            <div className="glass p-6 flex flex-col items-center justify-center min-h-[280px]">
              <Icon name="bar_chart" className="text-outline opacity-40 animate-pulse" style={{ fontSize: "48px" }} />
              <p className="text-3xl font-black text-primary font-mono mt-3">{formatScore(score, metric)}</p>
              <p className="text-sm font-bold text-on-surface-variant font-mono mt-1">{metricLabel(metric).long}</p>
            </div>
          )}

          <ResultPlotCard runId={runId} problemType={result.problem_type} />
        </motion.section>

        {/* Hyperparameter Tuning Before/After Results */}
        {extra.tuning_trials && extra.tuning_trials.length > 1 && (
          <motion.section
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.18 }}
          >
            <TuningResultsCard
              trials={extra.tuning_trials}
              baseline={extra.tuning_trials[0].score ?? score}
              optimized={score}
              metric={metric}
              modelName={modelName}
            />
          </motion.section>
        )}

        {/* Interactive Side-by-Side Model Comparison Grid */}
        {models.length > 0 && (
          <motion.section
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.24 }}
            className="glass overflow-hidden text-left select-none"
          >
            <div className="p-4 sm:p-5 border-b border-outline-variant flex flex-col sm:flex-row justify-between sm:items-center gap-1 bg-surface-container-low">
              <h3 className="text-lg sm:text-headline-md font-bold text-on-background">Model Comparison Leaderboard</h3>
              <span className="text-[10px] sm:text-xs font-bold text-on-surface-variant font-mono">CV 5-Fold metrics rank</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse font-sans text-xs sm:text-sm">
                <thead>
                  <tr className="bg-surface-container-low text-label-md text-on-surface-variant font-bold border-b border-outline-variant">
                    <th className="p-2 sm:p-5 text-[10px] sm:text-xs uppercase tracking-wider w-8 sm:w-20">#</th>
                    <th className="p-2 sm:p-5 text-[10px] sm:text-xs uppercase tracking-wider">Model</th>
                    <th className="p-2 sm:p-5 text-[10px] sm:text-xs uppercase tracking-wider text-center">CV Std</th>
                    <th className="p-2 sm:p-5 text-[10px] sm:text-xs uppercase tracking-wider text-center">CV Mean</th>
                    <th className="p-2 sm:p-5 text-[10px] sm:text-xs uppercase tracking-wider text-center hidden sm:table-cell">Test {metricLabel(metric).short}</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedModels.map((m, i) => {
                    const isWinner = m.name === modelName;
                    const testScore = isWinner ? extra.test_score ?? null : null;
                    return (
                      <tr
                        key={m.name}
                        className={cn(
                          "hover:bg-surface-container-low transition-colors border-b border-outline-variant/40 last:border-0"
                        )}
                      >
                        <td className={cn("p-2 sm:p-5 font-mono font-bold text-[11px] sm:text-sm", isWinner && "text-primary")}>{i + 1}</td>
                        <td className={cn("p-2 sm:p-5 font-mono font-bold", isWinner ? "text-primary" : "text-on-surface")}>
                          <div className="flex items-center gap-1 sm:gap-2">
                            <span className="text-[11px] sm:text-sm">{m.name}</span>
                          </div>
                        </td>
                        <td className={cn("p-2 sm:p-5 text-center font-mono text-[11px] sm:text-sm", isWinner ? "text-primary" : "text-on-surface-variant")}>± {(m.cv_std).toFixed(4)}</td>
                        <td className={cn("p-2 sm:p-5 text-center font-mono text-[11px] sm:text-sm", isWinner ? "text-primary" : "text-on-surface-variant")}>{formatScore(m.cv_mean, metric)}</td>
                        <td className={cn("p-2 sm:p-5 text-center font-mono hidden sm:table-cell", isWinner ? "text-primary" : "text-on-surface-variant")}>{testScore !== null ? formatScore(testScore, metric, 1) : "—"}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </motion.section>
        )}

      </div>
    </div>
  );
}
