"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useParams, useRouter } from "next/navigation";
import { startRun, getPreview, type StartRunBody } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Icon } from "@/components/icon";

interface PreviewData {
  run_id: string;
  filename?: string;
  source_format?: string;
  columns: string[];
  n_columns: number;
  preview: Record<string, unknown>[];
}

interface ColumnStats {
  column: string;
  type: "numeric" | "categorical";
  mean: string;
  std: string;
  min: string;
  max: string;
  nullPct: string;
  cardinality: number;
}

export default function PreviewPage() {
  const { runId } = useParams<{ runId: string }>();
  const router = useRouter();
  const [data, setData] = useState<PreviewData | null>(null);
  const [target, setTarget] = useState<string>("");
  // What to learn: predict a column, find groups (no target), or forecast over time.
  const [mode, setMode] = useState<"predict" | "cluster" | "forecast">("predict");
  const [dateColumn, setDateColumn] = useState<string>("");
  const [seriesColumn, setSeriesColumn] = useState<string>("");
  const [horizon, setHorizon] = useState<string>("");
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // DataTable States
  const [searchQuery, setSearchQuery] = useState("");
  const [currentPage, setCurrentPage] = useState(1);
  const [viewMode, setViewMode] = useState<"data" | "stats">("data");
  
  // Custom dropdown
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [dropdownRect, setDropdownRect] = useState<DOMRect | null>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  // Points at whichever responsive trigger (desktop or mobile) was last clicked.
  const triggerRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node) &&
          triggerRef.current && !triggerRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // While open, keep the fixed-position menu pinned to its trigger as the page
  // scrolls or the viewport resizes. Capture phase catches scrolls on inner
  // containers (scroll events don't bubble).
  useEffect(() => {
    if (!dropdownOpen) return;
    const reposition = () => {
      if (triggerRef.current) setDropdownRect(triggerRef.current.getBoundingClientRect());
    };
    window.addEventListener("scroll", reposition, true);
    window.addEventListener("resize", reposition);
    return () => {
      window.removeEventListener("scroll", reposition, true);
      window.removeEventListener("resize", reposition);
    };
  }, [dropdownOpen]);

  // Measure the button that was actually clicked — there are two responsive
  // triggers (desktop + mobile) and only one is visible at a time, so a single
  // shared ref would resolve to the hidden one and mis-position the menu.
  function openDropdown(e: React.MouseEvent<HTMLButtonElement>) {
    triggerRef.current = e.currentTarget;
    setDropdownRect(e.currentTarget.getBoundingClientRect());
    setDropdownOpen((o) => !o);
  }

  const pageSize = 10;

  useEffect(() => {
    getPreview(runId)
      .then((d) => {
        setData(d);
        if (d.columns?.length) setTarget(d.columns[d.columns.length - 1]);
      })
      .catch((err) => {
        setError(err.message || "Failed to load preview.");
      });
  }, [runId]);

  const isImages = data?.source_format === "images";
  const canStart = isImages || mode === "cluster" || (!!target && (mode === "predict" || !!dateColumn));

  function startBody(): StartRunBody {
    if (isImages) return { task: "image_classification" };
    if (mode === "cluster") return { task: "clustering" };
    if (mode === "forecast") {
      return {
        task: "forecasting",
        target,
        date_column: dateColumn,
        series_id_column: seriesColumn || undefined,
        horizon: horizon ? Number(horizon) : undefined,
      };
    }
    return { task: "auto", target };
  }

  async function handleStart() {
    if (!canStart) return;
    setStarting(true);
    setError(null);
    try {
      await startRun(runId, startBody());
      router.push(`/${runId}/running`);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to start run.");
      setStarting(false);
    }
  }

  if (!data) {
    return (
      <div className="flex-1 flex items-center justify-center select-none">
        <div className="flex flex-col items-center gap-3">
          <Icon name="sync" className="text-primary animate-spin" style={{ fontSize: "36px" }} />
          <p className="text-sm font-semibold text-on-surface-variant font-mono">Compiling dataset metrics...</p>
        </div>
      </div>
    );
  }

  const numericCols = data.columns.filter((col) => {
    const val = data.preview[0]?.[col];
    return typeof val === "number" || (typeof val === "string" && !isNaN(Number(val)) && val.trim() !== "");
  });
  const catCols = data.columns.filter((col) => !numericCols.includes(col));

  // Dynamic Summary Statistics Calculation
  const calculateSummaryStats = (): ColumnStats[] => {
    return data.columns.map((col) => {
      const isNum = numericCols.includes(col);
      const values = data.preview.map(row => row[col]);
      const validNums = values.map(v => Number(v)).filter(n => !isNaN(n) && n !== null && n !== undefined);
      
      const nullsCount = values.filter(v => v === null || v === undefined || v === "" || String(v).trim() === "-").length;
      const nullPct = ((nullsCount / values.length) * 100).toFixed(1);
      const uniqueVals = new Set(values.filter(v => v !== null && v !== undefined && v !== "")).size;

      if (isNum && validNums.length > 0) {
        const sum = validNums.reduce((a, b) => a + b, 0);
        const meanVal = sum / validNums.length;
        const minVal = Math.min(...validNums);
        const maxVal = Math.max(...validNums);
        
        // Std Dev
        const variance = validNums.reduce((a, b) => a + Math.pow(b - meanVal, 2), 0) / validNums.length;
        const stdVal = Math.sqrt(variance);

        return {
          column: col,
          type: "numeric",
          mean: meanVal.toFixed(2),
          std: stdVal.toFixed(2),
          min: minVal.toFixed(2),
          max: maxVal.toFixed(2),
          nullPct: `${nullPct}%`,
          cardinality: uniqueVals
        };
      } else {
        return {
          column: col,
          type: "categorical",
          mean: "N/A",
          std: "N/A",
          min: "N/A",
          max: "N/A",
          nullPct: `${nullPct}%`,
          cardinality: uniqueVals
        };
      }
    });
  };

  const summaryStats = calculateSummaryStats();

  // Real Data Health Metrics
  const nullRows = data.preview.filter(row =>
    data.columns.some(col => {
      const v = row[col];
      return v === null || v === undefined || String(v).trim() === "" || String(v).trim() === "-";
    })
  ).length;
  const totalCells = data.preview.length * data.columns.length;
  const nullCells = data.columns.reduce((acc, col) => {
    return acc + data.preview.filter(row => {
      const v = row[col];
      return v === null || v === undefined || String(v).trim() === "" || String(v).trim() === "-";
    }).length;
  }, 0);
  const healthScore = totalCells > 0 ? Math.round(((totalCells - nullCells) / totalCells) * 100) : 100;
  const healthColor = healthScore >= 90 ? "text-success-green" : healthScore >= 70 ? "text-warning-orange" : "text-error";
  const healthBarColor = healthScore >= 90 ? "bg-success-green" : healthScore >= 70 ? "bg-warning-orange" : "bg-error";

  // Search & Pagination Logic
  const filteredRows = data.preview.filter(row => {
    return data.columns.some(col => {
      const val = String(row[col] ?? "").toLowerCase();
      return val.includes(searchQuery.toLowerCase());
    });
  });

  const totalPages = Math.ceil(filteredRows.length / pageSize);
  const paginatedRows = filteredRows.slice((currentPage - 1) * pageSize, currentPage * pageSize);

  const numericPct = data.n_columns > 0 ? (numericCols.length / data.n_columns) * 100 : 0;
  const catPct = 100 - numericPct;

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4 md:p-gutter relative select-none">
      <div className="max-w-[1280px] mx-auto w-full">
        
        {/* Page Header */}
        <div className="mb-4 md:mb-8 flex flex-col md:flex-row md:items-end justify-between gap-2 md:gap-4">
          <div>
            <h1 className="text-xl md:text-headline-lg font-bold text-on-background mb-1 md:mb-2">
              Dataset: {data.filename ?? runId?.slice(0, 16)}
            </h1>
            <p className="text-sm md:text-body-lg text-on-surface-variant">
              Full feature breakdown, data quality metrics, and schema targets for training.
            </p>
          </div>
        </div>

        {/* Bento Grid with Gradients & Tooltips */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-3 md:gap-6 mb-4 md:mb-8">

          {/* Card 1: Target Prediction Variable */}
          <div className="col-span-1 lg:col-span-6 glass p-4 md:p-6 min-h-[160px] md:min-h-[220px] flex flex-col justify-between">
            <div className="flex items-start gap-3">
              <Icon name="target" className="text-primary bg-surface-purple-tint p-3 rounded-lg shrink-0" style={{ fontSize: "24px" }} />
              <div>
                <span className="text-xs font-bold uppercase tracking-wider text-primary mb-1 block">Prediction Goal</span>
                <h3 className="text-headline-md font-bold text-on-background">
                  Target Variable
                </h3>
              </div>
            </div>

            <div className="mt-4">
              <h2 className="text-2xl font-black text-primary font-mono break-words" title={target || "Awaiting target selection..."}>
                {target || "Select target..."}
              </h2>
              <p className="text-xs text-on-surface-variant mt-1.5">Selected column model will learn to predict.</p>
            </div>
          </div>

          {/* Card 2: Data Health Indicators */}
          <div className="col-span-1 lg:col-span-3 glass p-4 md:p-6 flex flex-col justify-between relative">
            <div className="flex items-start gap-3">
              <Icon name="health_and_safety" className="text-success-green bg-surface-green-tint p-3 rounded-lg shrink-0" style={{ fontSize: "24px" }} />
              <div>
                <span className="text-xs font-bold uppercase tracking-wider text-outline mb-1 block">Quality Scan</span>
                <h3 className="text-headline-md font-bold text-on-background">
                  Data Health
                </h3>
              </div>
            </div>

            <div className="mt-4">
              <div className="flex items-baseline gap-1.5 mb-2">
                <span className={cn("text-3xl font-black font-mono leading-none", healthColor)}>{healthScore}%</span>
                <span className="text-xs font-semibold text-on-surface-variant">Health Score</span>
              </div>
              <div className="h-2 w-full bg-surface-variant rounded-full overflow-hidden">
                <div className={cn("h-full rounded-full", healthBarColor)} style={{ width: `${healthScore}%` }} />
              </div>
              <div className="flex justify-between text-[11px] text-outline uppercase tracking-wider mt-3 font-bold font-mono">
                <span>{nullRows} Null {nullRows === 1 ? "row" : "rows"}</span>
                <span>{nullCells} Null {nullCells === 1 ? "cell" : "cells"}</span>
              </div>
            </div>
          </div>

          {/* Card 3: Feature Space Stacked bar */}
          <div className="col-span-1 lg:col-span-3 glass p-4 md:p-6 flex flex-col justify-between relative">
            <div className="flex items-start gap-3">
              <Icon name="category" className="text-info-blue bg-surface-purple-tint p-3 rounded-lg shrink-0" style={{ fontSize: "24px" }} />
              <div>
                <span className="text-xs font-bold uppercase tracking-wider text-outline mb-1 block">Dimension</span>
                <h3 className="text-headline-md font-bold text-on-background whitespace-nowrap">
                  Feature Space
                </h3>
              </div>
            </div>

            <div className="mt-4">
              <div className="flex justify-between items-baseline mb-2">
                <div>
                  <span className="text-3xl font-black text-info-blue font-mono leading-none">{data.n_columns}</span>
                  <span className="text-xs font-semibold text-on-surface-variant ml-1">Total Columns</span>
                </div>
              </div>
              
              {/* Stacked bar indicator */}
              <div className="h-2 w-full bg-surface-variant rounded-full overflow-hidden flex">
                <div className="h-full bg-info-blue" style={{ width: `${numericPct}%` }} title={`Numeric: ${numericCols.length}`} />
                <div className="h-full bg-warning-orange" style={{ width: `${catPct}%` }} title={`Categorical: ${catCols.length}`} />
              </div>
              
              <div className="flex justify-between text-[12px] text-outline uppercase tracking-wider mt-3 font-bold font-mono">
                <span className="text-info-blue">{numericCols.length} Numeric</span>
                <span className="text-warning-orange">{catCols.length} Categorical</span>
              </div>
            </div>
          </div>

        </div>

        {/* Target Column Selection + Run */}
        <div className="glass p-4 md:p-6 mb-8 flex flex-col gap-3">

          {/* Top row: title + run button (always a row) */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 shrink-0">
              <Icon name="target" className="text-primary" />
              <h3 className="text-headline-md font-bold text-on-surface whitespace-nowrap">
                {isImages ? "Image classification" : mode === "cluster" ? "Find groups" : mode === "forecast" ? "Forecast target" : "Target Variable"}
              </h3>
            </div>

            {/* Dropdown — hidden on mobile, fills space on desktop */}
            <div className={cn("relative flex-1 hidden", !isImages && mode !== "cluster" && "md:block")}>
              <button
                type="button"
                onClick={openDropdown}
                className="w-full bg-surface-variant border border-outline-variant rounded-lg py-2.5 px-3 text-sm text-on-surface font-semibold flex items-center justify-between gap-2 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all cursor-pointer"
              >
                <span>{target}</span>
                <Icon
                  name="expand_more"
                  className={cn("text-outline transition-transform duration-200", dropdownOpen && "rotate-180")}
                  style={{ fontSize: "16px" }}
                />
              </button>
            </div>

            {/* Run button — pushed to far right */}
            <div className="flex items-center gap-3 shrink-0 ml-auto md:ml-0">
              {error && <p className="text-xs text-error font-medium">{error}</p>}
              <button
                onClick={handleStart}
                disabled={!canStart || starting}
                className={cn(
                  "btn-primary text-label-md px-4 md:px-6 py-3 rounded-lg",
                  (!canStart || starting) && "opacity-50 cursor-not-allowed"
                )}
              >
                <span className="hidden sm:inline">{starting ? "Running..." : "Run"}</span>
                <span className="sm:hidden">{!starting && "Run"}</span>
                <Icon name={starting ? "sync" : "arrow_forward"} className={cn(starting && "animate-spin")} style={{ fontSize: "16px" }} />
              </button>
            </div>
          </div>

          {/* Task switch: images have one task, tables three */}
          {isImages ? (
            <p className="text-xs text-on-surface-variant">
              Each folder in the zip is a class; pretrained CNNs are fine-tuned on the GPU to tell them apart.
            </p>
          ) : (
            <div className="flex border border-outline-variant rounded-lg p-0.5 bg-surface-container-low w-full sm:w-fit text-xs font-bold">
              {([["predict", "Predict a column"], ["cluster", "Find groups"], ["forecast", "Forecast over time"]] as const).map(([key, label]) => (
                <button
                  key={key}
                  type="button"
                  onClick={() => setMode(key)}
                  className={cn(
                    "flex-1 sm:flex-none px-3 py-1.5 rounded transition-colors",
                    mode === key ? "bg-surface-container-lowest text-primary shadow-sm" : "text-outline hover:text-primary"
                  )}
                >
                  {label}
                </button>
              ))}
            </div>
          )}

          {mode === "cluster" && !isImages && (
            <p className="text-xs text-on-surface-variant">
              No target: every column is used to find natural groups (K-Means, HDBSCAN, DBSCAN, Spectral), each row gets a cluster.
            </p>
          )}

          {mode === "forecast" && !isImages && (
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-xs font-semibold">
              <label className="flex flex-col gap-1 text-outline">
                Date column
                <select value={dateColumn} onChange={(e) => setDateColumn(e.target.value)}
                  className="bg-surface-variant border border-outline-variant rounded-lg py-2 px-2 text-sm text-on-surface">
                  <option value="">Choose...</option>
                  {data.columns.filter((c) => c !== target).map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-outline">
                Series id (optional)
                <select value={seriesColumn} onChange={(e) => setSeriesColumn(e.target.value)}
                  className="bg-surface-variant border border-outline-variant rounded-lg py-2 px-2 text-sm text-on-surface">
                  <option value="">One series</option>
                  {data.columns.filter((c) => c !== target && c !== dateColumn).map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
              <label className="flex flex-col gap-1 text-outline">
                Horizon (steps, optional)
                <input type="number" min={1} max={1000} value={horizon} onChange={(e) => setHorizon(e.target.value)}
                  placeholder="Auto from frequency"
                  className="bg-surface-variant border border-outline-variant rounded-lg py-2 px-2 text-sm text-on-surface" />
              </label>
            </div>
          )}

          {/* Dropdown — full width on mobile only */}
          <div className={cn("relative md:hidden", (isImages || mode === "cluster") && "hidden")}>
            <button
              type="button"
              onClick={openDropdown}
              className="w-full bg-surface-variant border border-outline-variant rounded-lg py-2.5 px-3 text-sm text-on-surface font-semibold flex items-center justify-between gap-2 focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all cursor-pointer"
            >
              <span>{target}</span>
              <Icon
                name="expand_more"
                className={cn("text-outline transition-transform duration-200", dropdownOpen && "rotate-180")}
                style={{ fontSize: "16px" }}
              />
            </button>
          </div>

          {/* Portal dropdown list — shared for both triggers */}
          {dropdownOpen && dropdownRect && createPortal(
            <div
              ref={dropdownRef}
              style={{
                position: "fixed",
                top: dropdownRect.bottom + 4,
                left: dropdownRect.left,
                width: dropdownRect.width,
                zIndex: 9999,
                backgroundColor: "#0e0e16",
              }}
              className="rounded-lg border border-outline-variant overflow-auto max-h-60 shadow-lg"
            >
              <ul>
                {data.columns.map((col) => (
                  <li
                    key={col}
                    onClick={() => { setTarget(col); setDropdownOpen(false); }}
                    className={cn(
                      "px-3 py-2 text-sm font-semibold cursor-pointer transition-colors",
                      col === target
                        ? "text-primary bg-surface-purple-tint"
                        : "text-on-surface hover:bg-surface-variant"
                    )}
                  >
                    {col}
                  </li>
                ))}
              </ul>
            </div>,
            document.body
          )}
        </div>

        {/* Data Table with Raw/Stats Toggle & Search */}
        <div className="glass overflow-hidden">

          {/* Table Toolbar controls */}
          <div className="p-5 border-b border-outline-variant flex flex-col gap-3 sm:grid sm:grid-cols-3 sm:items-center sm:gap-4 bg-surface-container-low">
            <h3 className="text-headline-md font-bold text-on-background">
              {viewMode === "data" ? `Dataset Preview - ${data.preview.length.toLocaleString()} rows` : "Features Profiling"}
            </h3>

            {/* Toggle + Search in one row on mobile; separate grid columns on sm+ */}
            <div className="flex items-center gap-3 sm:contents">
              <div className="flex flex-1 sm:flex-none sm:justify-center">
                <div className="flex border border-outline-variant rounded-lg p-0.5 bg-surface-container-low select-none w-full">
                  <button
                    onClick={() => setViewMode("data")}
                    className={cn(
                      "flex-1 px-3 py-1.5 rounded-md text-xs font-bold transition-all text-center",
                      viewMode === "data" ? "bg-surface-container-lowest text-primary shadow-sm" : "text-on-surface-variant hover:text-primary"
                    )}
                  >
                    Raw Data
                  </button>
                  <button
                    onClick={() => setViewMode("stats")}
                    className={cn(
                      "flex-1 px-3 py-1.5 rounded-md text-xs font-bold transition-all text-center",
                      viewMode === "stats" ? "bg-surface-container-lowest text-primary shadow-sm" : "text-on-surface-variant hover:text-primary"
                    )}
                  >
                    Summary Stats
                  </button>
                </div>
              </div>

              {/* Row Search Filter — right-aligned, only in Data view */}
              <div className="hidden sm:flex justify-end sm:flex-1">
              {viewMode === "data" && (
                <div className="relative w-40 md:w-56">
                  <Icon name="search" className="absolute left-3 top-1/2 -translate-y-1/2 text-outline" style={{ fontSize: "15px" }} />
                  <input
                    className="w-full bg-surface-variant border border-outline-variant rounded-lg py-1.5 pl-9 pr-4 text-xs text-on-surface focus:outline-none focus:border-primary focus:ring-1 focus:ring-primary transition-all"
                    placeholder="Filter rows..."
                    type="text"
                    value={searchQuery}
                    onChange={(e) => { setSearchQuery(e.target.value); setCurrentPage(1); }}
                  />
                </div>
              )}
              </div>
            </div>
          </div>

          <div className="overflow-auto max-h-[600px] border-b border-outline-variant">
            {viewMode === "data" ? (
              // RAW DATATABLE PREVIEW
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-surface-container-low text-label-md text-on-surface-variant font-bold border-b border-outline-variant">
                    {data.columns.map((col) => (
                      <th
                        key={col}
                        className={cn(
                          "p-4 font-bold whitespace-nowrap text-xs uppercase tracking-wider",
                          col === target ? "text-primary bg-surface-purple-tint/30 border-x border-primary/10" : ""
                        )}
                      >
                        <div className="flex items-center gap-1">
                          <Icon name={numericCols.includes(col) ? "tag" : "text_fields"} className="text-[12px]" />
                          {col}
                        </div>
                        {col === target && (
                          <span className="ml-1.5 text-[8px] bg-primary text-on-primary px-1.5 py-0.5 rounded-full font-sans uppercase tracking-widest font-black">
                            Target
                          </span>
                        )}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="text-body-md text-on-background font-mono text-xs">
                  {filteredRows.length > 0 ? (
                    filteredRows.map((row, i) => (
                      <tr
                        key={i}
                        className="hover:bg-surface-container-low transition-colors border-b border-outline-variant/40 last:border-0"
                      >
                        {data.columns.map((col) => (
                          <td
                            key={col}
                            className={cn(
                              "p-4 whitespace-nowrap",
                              col === target ? "text-primary font-bold bg-surface-purple-tint/10 border-x border-primary/5" : ""
                            )}
                          >
                            {String(row[col] ?? "-")}
                          </td>
                        ))}
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={data.columns.length} className="p-8 text-center text-on-surface-variant font-sans text-xs">
                        No rows matching the filter were found.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            ) : (
              // SUMMARY STATISTICS TABLE
              <table className="w-full text-left border-collapse font-sans">
                <thead>
                  <tr className="bg-surface-container-low text-label-md text-on-surface-variant font-bold border-b border-outline-variant">
                    <th className="p-4 text-xs uppercase tracking-wider">Column Feature</th>
                    <th className="p-4 text-xs uppercase tracking-wider">Semantic Type</th>
                    <th className="p-4 text-xs uppercase tracking-wider text-right font-mono">Mean</th>
                    <th className="p-4 text-xs uppercase tracking-wider text-right font-mono">Std Dev</th>
                    <th className="p-4 text-xs uppercase tracking-wider text-right font-mono">Min</th>
                    <th className="p-4 text-xs uppercase tracking-wider text-right font-mono">Max</th>
                    <th className="p-4 text-xs uppercase tracking-wider text-right font-mono">Null %</th>
                    <th className="p-4 text-xs uppercase tracking-wider text-right font-mono">Cardinality</th>
                  </tr>
                </thead>
                <tbody className="text-xs text-on-background">
                  {summaryStats.map((stat) => (
                    <tr
                      key={stat.column}
                      className={cn(
                        "hover:bg-surface-container-low transition-colors border-b border-outline-variant/40 last:border-0",
                        stat.column === target ? "bg-surface-purple-tint/10" : ""
                      )}
                    >
                      <td className="p-4 font-bold text-on-surface flex items-center gap-1.5">
                        <Icon name={stat.type === "numeric" ? "tag" : "text_fields"} className="text-[12px]" />
                        {stat.column}
                        {stat.column === target && (
                          <span className="text-[8px] bg-primary text-on-primary px-1.5 py-0.5 rounded-full font-sans uppercase font-black tracking-widest shrink-0 ml-1">Target</span>
                        )}
                      </td>
                      <td className="p-4">
                        <span className={cn(
                          "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider",
                          stat.type === "numeric" ? "bg-surface-purple-tint text-primary" : "bg-secondary-container/20 text-secondary"
                        )}>
                          {stat.type}
                        </span>
                      </td>
                      <td className="p-4 text-right font-mono text-on-surface-variant font-medium">{stat.mean}</td>
                      <td className="p-4 text-right font-mono text-on-surface-variant font-medium">{stat.std}</td>
                      <td className="p-4 text-right font-mono text-on-surface-variant font-medium">{stat.min}</td>
                      <td className="p-4 text-right font-mono text-on-surface-variant font-medium">{stat.max}</td>
                      <td className="p-4 text-right font-mono text-error font-bold">{stat.nullPct}</td>
                      <td className="p-4 text-right font-mono font-bold text-on-surface">{stat.cardinality}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

        </div>
      </div>
    </div>
  );
}
