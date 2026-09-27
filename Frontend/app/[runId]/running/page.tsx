"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { motion, AnimatePresence } from "framer-motion";
import { cancelRun, getStatus, getDiagnostics, type Diagnostics } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Icon } from "@/components/icon";

const STEPS = [
  {
    key: "profile",
    fn: "profile_dataset(run_id)",
    label: "Profile Dataset",
    icon: "database",
    code: `import pandas as pd
import numpy as np

# Load dataset and profile basic statistics
df = pd.read_csv("storage/runs/{run_id}/dataset.csv")
null_counts = df.isnull().sum()
dtypes = df.dtypes

print(f"Loaded {len(df)} rows across {len(df.columns)} columns.")
print("Missing values mapped.")`,
    outputLines: [
      "[SYSTEM] Initializing reader buffers...",
      "[INFO] Reading column metadata & binary bytes...",
      "[INFO] Scanning dtypes and distinct cardinality counts",
      "[INFO] Evaluating missing values frequency per feature",
      "[INFO] Collecting 5-row sample matrix for schema verification",
      "[SUCCESS] Profiling complete. Dataset parsed successfully."
    ],
  },
  {
    key: "detect",
    fn: "detect_problem_type(run_id, target)",
    label: "Detect Problem Type",
    icon: "target",
    code: `# Determine ML objective (Classification vs Regression)
target_col = df[target]
cardinality = target_col.nunique()

if target_col.dtype == 'object' or cardinality < 15:
    problem_type = "classification"
else:
    problem_type = "regression"

print(f"Target variable has {cardinality} classes. Selected: {problem_type}")`,
    outputLines: [
      "[SYSTEM] Fetching configured target column...",
      "[INFO] Inspecting target column data type and structure",
      "[INFO] Counting unique values in target labels",
      "[INFO] Evaluating cardinality threshold limit (threshold=15)",
      "[SUCCESS] Problem type successfully determined."
    ],
  },
  {
    key: "eda",
    fn: "run_eda(run_id, target)",
    label: "Explore the Data",
    icon: "bar_chart",
    code: `# Run Automated Exploratory Data Analysis
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corrwith(df[target])

print("Top 5 Pearson correlations with target:")
print(correlations.sort_values(ascending=False).head(5))`,
    outputLines: [
      "[SYSTEM] Launching analytical profiler...",
      "[INFO] Computing numeric summaries (mean, std, min, max)",
      "[INFO] Analyzing target distribution & class balancing",
      "[INFO] Computing Pearson correlations with target label",
      "[INFO] Identifying top features with highest mutual correlation",
      "[SUCCESS] EDA statistical report saved successfully."
    ],
  },
  {
    key: "feature_engineer",
    fn: "feature_engineer(run_id, target)",
    label: "Engineer Features",
    icon: "build",
    code: `from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Clean missing values and encode text values
imputer = SimpleImputer(strategy='median')
encoder = OneHotEncoder(handle_unknown='ignore')

# Drop columns with > 50% nulls
df_clean = df.drop(columns=[col for col in df if df[col].isnull().mean() > 0.5])`,
    outputLines: [
      "[SYSTEM] Booting preprocessing transformers...",
      "[INFO] Dropping columns with high rate (>50%) missing values",
      "[INFO] Removing ID-like features (cardinality metric ≥95% unique)",
      "[INFO] Applying One-Hot encoding to low-cardinality categoricals",
      "[INFO] Applying Label-encoding to high-cardinality categoricals",
      "[SUCCESS] Engineered feature space saved to storage node."
    ],
  },
  {
    key: "train",
    fn: "train_model(run_id, target, problem_type)",
    label: "Train & Compare Models",
    icon: "model_training",
    code: `from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

# Compile model candidates and score via 5-Fold Cross Validation
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=200, random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000)
}

scores = {name: cross_val_score(model, X, y, cv=5).mean() for name, model in models.items()}`,
    outputLines: [
      "[SYSTEM] Allocating model candidates memory cache...",
      "[INFO] Splitting dataset → 80% train / 20% test subsets",
      "[INFO] Fitting Imputation Imputer on training samples only",
      "[INFO] Training RandomForestClassifier (n_estimators=200)...",
      "[INFO] Training ExtraTreesClassifier (n_estimators=200)...",
      "[INFO] Training GradientBoostingClassifier (early stopping enabled)...",
      "[INFO] Training LogisticRegression estimator...",
      "[INFO] Evaluating 5-Fold cross-validation scores on all estimators",
      "[INFO] Selecting champion architecture by Cross-Validation mean",
      "[SUCCESS] Training phase completed. Best model compiled."
    ],
  },
  {
    key: "visualize",
    fn: "generate_visualization(run_id, target, problem_type)",
    label: "Generate Results",
    icon: "insights",
    code: `import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Compute evaluation plots and save matrix assets
y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Purples)
plt.savefig("storage/runs/{run_id}/plot.png")`,
    outputLines: [
      "[SYSTEM] Rendering performance engine...",
      "[INFO] Loading held-out test predictions matrices",
      "[INFO] Computing Confusion Matrix elements & error residuals",
      "[INFO] Rendering plot visualizations to PNG asset",
      "[SUCCESS] Performance plot successfully saved to run directory."
    ],
  },
];

type StepStatus = "waiting" | "active" | "done" | "error";

// Sidebar run diagnostics: recorded state only. Training runs on separate GPU
// containers, so there are no live utilisation numbers to show here.
function ServerDiagnosticsWidget({ runId }: { runId: string }) {
  const [diag, setDiag] = useState<Diagnostics | null>(null);

  useEffect(() => {
    const fetchDiagnostics = async () => {
      try {
        setDiag(await getDiagnostics(runId));
      } catch (err) {
        console.error("Failed to fetch diagnostics:", err);
      }
    };

    fetchDiagnostics();
    const timer = setInterval(fetchDiagnostics, 3000);
    return () => clearInterval(timer);
  }, [runId]);

  const rows: [string, string][] = [
    ["Stage", diag?.stage ? diag.stage.replace(/_/g, " ") : "waiting to start"],
    ["Task", diag?.task ? diag.task.replace(/_/g, " ") : "-"],
    ["Training on", diag?.gpu ?? "-"],
    ["GPU containers", diag ? String(diag.gpu_calls) : "-"],
    ["Elapsed", diag?.elapsed_seconds != null ? `${Math.floor(diag.elapsed_seconds / 60)}m ${diag.elapsed_seconds % 60}s` : "-"],
  ];

  return (
    <div className="bg-surface-container border border-outline-variant rounded-xl p-4 font-sans space-y-3 shadow-sm select-none">
      <div className="flex items-center gap-1.5 border-b border-outline-variant pb-2">
        <span className="w-2 h-2 rounded-full bg-success-green animate-pulse" />
        <h4 className="text-[10px] font-bold text-outline uppercase tracking-wider">Run Diagnostics</h4>
      </div>
      <div className="space-y-1.5">
        {rows.map(([label, value]) => (
          <div key={label} className="flex justify-between gap-3 text-[10px] font-semibold text-on-surface">
            <span>{label}</span>
            <span className="font-mono text-primary text-right truncate">{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function SidebarStep({
  step,
  index,
  status,
  isLast,
}: {
  step: (typeof STEPS)[number];
  index: number;
  status: StepStatus;
  isLast: boolean;
}) {
  return (
    <div className="relative flex gap-2.5">
      {!isLast && (
        <div
          className={cn(
            "absolute left-[12px] top-[26px] w-px bottom-0",
            status === "done" ? "bg-success-green/40" : "bg-outline-variant"
          )}
        />
      )}

      <div className="shrink-0 z-10 mt-0.5">
        {status === "done" ? (
          <div className="w-[25px] h-[25px] rounded-full bg-surface-green-tint border border-success-green flex items-center justify-center">
            <Icon name="check_circle" className="text-success-green" style={{ fontSize: "13px" }} />
          </div>
        ) : status === "active" ? (
          <div className="w-[25px] h-[25px] rounded-full bg-surface-purple-tint border border-primary flex items-center justify-center">
            <motion.span
              animate={{ scale: [1, 1.15, 1] }}
              transition={{ duration: 1.2, repeat: Infinity }}
              className="text-primary inline-flex"
              style={{ fontSize: "12px" }}
            >
              <Icon name={step.icon} />
            </motion.span>
          </div>
        ) : status === "error" ? (
          <div className="w-[25px] h-[25px] rounded-full bg-error-container border border-error flex items-center justify-center">
            <Icon name="error" className="text-error" style={{ fontSize: "12px" }} />
          </div>
        ) : (
          <div className="w-[25px] h-[25px] rounded-full bg-surface-container border border-outline-variant flex items-center justify-center">
            <span className="text-[9px] font-bold text-outline">{index + 1}</span>
          </div>
        )}
      </div>

      <div className={cn("flex-1 min-w-0", !isLast && "pb-3.5")}>
        <p
          className={cn(
            "text-label-md leading-tight py-1 truncate font-medium",
            status === "active" && "text-primary font-bold",
            status === "done" && "text-on-surface-variant",
            status === "waiting" && "text-outline opacity-60",
            status === "error" && "text-error"
          )}
        >
          {step.label}
        </p>
      </div>
    </div>
  );
}

function PipelineCell({
  step,
  index,
  status,
  cellNumber,
  elapsedAtDone,
  expanded,
  onToggle,
  logFilter,
}: {
  step: (typeof STEPS)[number];
  index: number;
  status: StepStatus;
  cellNumber: number;
  elapsedAtDone?: number;
  expanded: boolean;
  onToggle: () => void;
  logFilter: "all" | "success" | "system";
}) {
  const [visibleLines, setVisibleLines] = useState(0);
  const [codeOpen, setCodeOpen] = useState(false);

  useEffect(() => {
    if (status !== "active") { setVisibleLines(0); return; }
    let i = 0;
    const tick = () => {
      i++;
      setVisibleLines(i);
      if (i < step.outputLines.length) setTimeout(tick, 300 + Math.random() * 200);
    };
    setTimeout(tick, 200);
  }, [status, step.outputLines.length]);

  const isClickable = status === "done" || status === "error";
  const showOutput = status === "active" || (isClickable && expanded);

  const counterLabel =
    status === "waiting" ? " " :
    status === "active" ? "*" :
    status === "done" ? String(cellNumber) : "!";

  // Filter output lines
  const filterLine = (line: string): boolean => {
    if (logFilter === "all") return true;
    if (logFilter === "success") return line.startsWith("[SUCCESS]");
    if (logFilter === "system") return line.startsWith("[SYSTEM]");
    return true;
  };

  const activeFilteredLines = step.outputLines.slice(0, visibleLines).filter(filterLine);
  const doneFilteredLines = step.outputLines.filter(filterLine);

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: status === "waiting" ? 0.38 : 1, y: 0 }}
      transition={{ delay: index * 0.03, duration: 0.22 }}
      className={cn(
        "bg-surface-container-lowest border rounded-xl overflow-hidden shadow-sm",
        status === "active" && "border-primary/40",
        status === "done" && "border-success-green/40",
        status === "error" && "border-error/40",
        status === "waiting" && "border-outline-variant"
      )}
    >
      {/* Header */}
      <div
        className={cn(
          "flex items-center gap-1.5 px-3 py-1.5 sm:px-3.5 sm:py-2 border-b select-none",
          status === "active" && "bg-surface-purple-tint/50 border-primary/20",
          status === "done" && "bg-surface-green-tint/30 border-success-green/20",
          status === "error" && "bg-error-container/30 border-error/20",
          status === "waiting" && "bg-surface-container border-outline-variant"
        )}
      >
        <span
          className={cn(
            "text-[10px] font-mono shrink-0 w-7 select-none",
            status === "active" ? "text-primary" :
            status === "done" ? "text-success-green" :
            status === "error" ? "text-error" : "text-outline"
          )}
        >
          In [{counterLabel}]:
        </span>

        {status === "active" && (
          <motion.div
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
            className="w-3 h-3 rounded-full border-[1.5px] border-primary border-t-transparent shrink-0"
          />
        )}
        {status === "done" && (
          <Icon name="check_circle" className="text-success-green shrink-0 animate-scale" style={{ fontSize: "14px" }} />
        )}
        {status === "error" && (
          <Icon name="error" className="text-error shrink-0 animate-scale" style={{ fontSize: "14px" }} />
        )}

        <span className="truncate flex-1 font-mono text-xs text-on-surface">
          <span className="hidden sm:inline text-primary font-bold">{step.fn.split("(")[0]}</span>
          <span className="text-outline">(</span>
          <span className="text-warning-orange font-semibold">{step.fn.split("(")[1]?.replace(")", "")}</span>
          <span className="text-outline">)</span>
        </span>

        {/* Expanders and Code toggle actions */}
        <div className="ml-auto flex flex-row items-center gap-1 sm:gap-3 shrink-0">
          {status !== "waiting" && (
            <button
              onClick={() => setCodeOpen(!codeOpen)}
              className={cn(
                "text-[10px] font-mono border px-1.5 py-0.5 sm:px-2 rounded transition-all flex items-center gap-1 hover:border-primary hover:text-primary",
                codeOpen ? "bg-surface-purple-tint border-primary/30 text-primary font-semibold" : "border-outline-variant text-outline"
              )}
            >
              <Icon name="code" style={{ fontSize: "12px" }} />
              <span className="hidden sm:inline">{codeOpen ? "Hide Code" : "Show Code"}</span>
            </button>
          )}

          {status === "done" && elapsedAtDone !== undefined && (
            <span className="text-[10px] text-on-surface-variant font-mono font-bold">{elapsedAtDone}s</span>
          )}
          {isClickable && (
            <button onClick={onToggle} aria-label={expanded ? "Collapse output" : "Expand output"} className="text-outline hover:text-primary p-0.5 rounded hover:bg-surface-container">
              <Icon name={expanded ? "expand_less" : "expand_more"} className="block" style={{ fontSize: "16px" }} />
            </button>
          )}
        </div>
      </div>

      {/* Code panel editor */}
      <AnimatePresence>
        {codeOpen && (
          <motion.div
            initial={{ height: 0 }}
            animate={{ height: "auto" }}
            exit={{ height: 0 }}
            className="bg-surface-container border-b border-outline-variant overflow-hidden font-mono text-xs text-left"
          >
            <div className="p-3 bg-[#1e1e24] text-slate-300 relative">
              <span className="absolute top-2 right-3 text-[9px] font-bold text-slate-500 uppercase font-sans">Python Executable</span>
              <pre className="overflow-x-auto whitespace-pre leading-relaxed select-all">
                {step.code}
              </pre>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Output Console Log Panel */}
      <AnimatePresence initial={false}>
        {showOutput && (
          <motion.div
            key="output"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.18, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="px-3.5 py-2.5 space-y-1 bg-surface-container-low font-mono text-[11px] text-left border-t border-outline-variant/30">
              {status === "done"
                ? doneFilteredLines.map((line, i) => {
                    const isSuccess = line.startsWith("[SUCCESS]");
                    const isSystem = line.startsWith("[SYSTEM]");
                    const text = line.replace(/^\[[A-Z]+\]\s*/, "");
                    return (
                      <p key={i} className="text-on-surface-variant leading-relaxed">
                        <span className={cn(
                          "font-bold mr-1.5",
                          isSuccess ? "text-success-green" : isSystem ? "text-primary" : "text-info-blue"
                        )}>
                          {isSuccess ? "✓ [SUCCESS]" : isSystem ? "⚙ [SYSTEM]" : "ℹ [INFO]"}
                        </span>
                        {text}
                      </p>
                    );
                  })
                : activeFilteredLines.map((line, i) => {
                    const isSuccess = line.startsWith("[SUCCESS]");
                    const isSystem = line.startsWith("[SYSTEM]");
                    const text = line.replace(/^\[[A-Z]+\]\s*/, "");
                    return (
                      <motion.p
                        key={i}
                        initial={{ opacity: 0, x: -3 }}
                        animate={{ opacity: 1, x: 0 }}
                        className="text-on-surface-variant leading-relaxed"
                      >
                        <span className={cn(
                          "font-bold mr-1.5",
                          isSuccess ? "text-success-green" : isSystem ? "text-primary" : "text-info-blue"
                        )}>
                          {isSuccess ? "✓ [SUCCESS]" : isSystem ? "⚙ [SYSTEM]" : "ℹ [INFO]"}
                        </span>
                        {text}
                        {i === activeFilteredLines.length - 1 && (
                          <span className="inline-block w-1.5 h-3.5 bg-primary ml-0.5 animate-blink align-middle rounded-sm" />
                        )}
                      </motion.p>
                    );
                  })}
              
              {status === "done" && doneFilteredLines.length === 0 && (
                <p className="text-outline text-center py-2">No logs matching selected level filter.</p>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export default function RunningPage() {
  const { runId } = useParams<{ runId: string }>();
  const router = useRouter();

  const [activeStep, setActiveStep] = useState(0);
  const [failed, setFailed] = useState(false);
  const [errMsg, setErrMsg] = useState("");
  const [elapsed, setElapsed] = useState(0);
  const [stepElapsed, setStepElapsed] = useState<number[]>([]);
  const [expandedCells, setExpandedCells] = useState<Set<number>>(new Set());
  const [isComplete, setIsComplete] = useState(false);

  // Custom Log filter
  const [logFilter, setLogFilter] = useState<"all" | "success" | "system">("all");

  const stepStart = useRef<number>(Date.now());
  const runStart = useRef<number>(Date.now());

  useEffect(() => {
    if (isComplete || failed) return;
    const id = setInterval(
      () => setElapsed(Math.floor((Date.now() - runStart.current) / 1000)),
      1000
    );
    return () => clearInterval(id);
  }, [isComplete, failed]);

  useEffect(() => {
    setExpandedCells((prev) => {
      const next = new Set(prev);
      next.add(activeStep);
      if (activeStep > 0) {
        next.delete(activeStep - 1);
      }
      return next;
    });
  }, [activeStep]);

  const toggleCell = (index: number) => {
    setExpandedCells((prev) => {
      const next = new Set(prev);
      if (next.has(index)) next.delete(index);
      else next.add(index);
      return next;
    });
  };

  useEffect(() => {
    let stepTimer: ReturnType<typeof setTimeout>;
    let poll: ReturnType<typeof setInterval>;
    const STEP_DURATIONS = [3200, 2800, 3600, 3000, 7000, 2500];

    const advanceStep = () => {
      setActiveStep((s) => {
        const el = Math.floor((Date.now() - stepStart.current) / 1000);
        setStepElapsed((prev) => { const n = [...prev]; n[s] = el; return n; });
        stepStart.current = Date.now();
        if (s < STEPS.length - 1) {
          stepTimer = setTimeout(advanceStep, STEP_DURATIONS[s + 1] + Math.random() * 1500);
          return s + 1;
        }
        return s;
      });
    };
    stepStart.current = Date.now();
    stepTimer = setTimeout(advanceStep, STEP_DURATIONS[0] + Math.random() * 1500);

    poll = setInterval(async () => {
      try {
        const s = await getStatus(runId);
        if (s.status === "succeeded") {
          clearInterval(poll); clearTimeout(stepTimer);
          setActiveStep(STEPS.length);
          setIsComplete(true);
        } else if (s.status === "failed" || s.status === "cancelled") {
          clearInterval(poll); clearTimeout(stepTimer);
          setFailed(true);
          setErrMsg(s.error ?? "Unknown error.");
        }
      } catch { /* keep polling */ }
    }, 3000);

    return () => { clearInterval(poll); clearTimeout(stepTimer); };
  }, [runId, router]);

  const allDone = activeStep >= STEPS.length;
  const progressPct = Math.min(100, (activeStep / STEPS.length) * 100);

  return (
    <div className="flex-1 flex flex-row overflow-hidden select-none relative">

      {/* Left pipeline progress panel sidebar — completely detached, never scrolls with cells */}
      <motion.aside
        initial={{ opacity: 0, x: -16 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.35 }}
        className="w-72 glass-strong border-r border-outline-variant px-5 py-6 hidden md:flex flex-col shrink-0 justify-between gap-6 absolute left-0 top-0 bottom-0 h-full overflow-y-auto z-10"
      >
        <div className="flex flex-col gap-5">
          {/* Header Progress status */}
          <div className="shrink-0">
            <div className="flex items-center gap-2 mb-1">
              <Icon name="model_training" className="text-primary" style={{ fontSize: "18px" }} />
              <h2 className="text-xs font-bold text-primary uppercase tracking-widest">Pipeline</h2>
            </div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold text-on-surface-variant font-sans">
                {allDone ? "Complete" : failed ? "Failed" : `Step ${Math.min(activeStep + 1, STEPS.length)} of ${STEPS.length}`}
              </span>
              <span className="text-xs font-bold text-on-surface-variant font-mono tabular-nums bg-surface-container-low px-2 py-0.5 rounded-full">{elapsed}s</span>
            </div>
            <div className="w-full bg-surface-variant rounded-full h-2 overflow-hidden">
              <motion.div
                className="h-full bg-primary-container rounded-full"
                animate={{ width: `${progressPct}%` }}
                transition={{ duration: 0.6, ease: "easeOut" }}
              />
            </div>
          </div>

          {/* Run diagnostics widget */}
          <ServerDiagnosticsWidget runId={runId} />
          {!isComplete && !failed && (
            <button
              onClick={async () => {
                try {
                  await cancelRun(runId);
                } catch (e) {
                  setErrMsg(e instanceof Error ? e.message : "Could not cancel the run.");
                }
              }}
              className="text-[11px] text-error font-bold hover:underline text-left"
            >
              Cancel run
            </button>
          )}

          {/* Staggered steps progress list */}
          <div className="space-y-1 flex-1">
            {STEPS.map((step, i) => {
              const status: StepStatus =
                failed && activeStep === i ? "error" :
                activeStep > i ? "done" :
                activeStep === i ? "active" : "waiting";
              return (
                <SidebarStep key={step.key} step={step} index={i} status={status} isLast={i === STEPS.length - 1} />
              );
            })}
          </div>
        </div>

        {failed && (
          <div className="bg-error-container border border-error/20 rounded-xl px-4 py-3 space-y-1.5 text-left font-sans animate-scale">
            <p className="text-xs font-bold text-error">{errMsg === "Cancelled by the user." ? "Run cancelled" : "Run execution failed"}</p>
            <p className="text-[11px] text-on-surface-variant leading-relaxed">{errMsg}</p>
            <button
              onClick={() => router.push("/")}
              className="text-[11px] text-primary font-bold hover:underline"
            >
              ← Start over ingestion
            </button>
          </div>
        )}
      </motion.aside>

      {/* Main Notebook Console — scrolls independently from the pipeline panel */}
      <div className="flex-1 flex flex-col overflow-hidden p-4 lg:p-6 ml-0 md:ml-72">
        <div className="max-w-2xl mx-auto w-full flex-1 flex flex-col gap-3 overflow-hidden">
          
          {/* Jupyter Notebook top toolbar bar */}
          <div className="flex items-center gap-3 glass rounded-xl px-4 py-2.5 shrink-0 select-none">
            <div className="flex gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-error-pink/70" />
              <div className="w-2.5 h-2.5 rounded-full bg-warning-orange/70" />
              <div className="w-2.5 h-2.5 rounded-full bg-success-green/70" />
            </div>
            
            <span className="text-xs font-bold text-on-surface-variant font-mono truncate flex-1 text-left ml-2 hidden sm:block">
              automl_forge_{runId?.slice(0, 8)}.ipynb
            </span>

            {/* Log filter toggles */}
            <div className="hidden sm:flex border border-outline-variant rounded-lg p-0.5 bg-surface-container-low shrink-0 text-[10px] font-bold">
              <button
                onClick={() => setLogFilter("all")}
                className={cn(
                  "px-2 py-1 rounded transition-colors",
                  logFilter === "all" ? "bg-surface-container-lowest text-primary shadow-sm" : "text-outline hover:text-primary"
                )}
              >
                ALL
              </button>
              <button
                onClick={() => setLogFilter("success")}
                className={cn(
                  "px-2 py-1 rounded transition-colors",
                  logFilter === "success" ? "bg-surface-container-lowest text-primary shadow-sm" : "text-outline hover:text-primary"
                )}
              >
                SUCCESS
              </button>
              <button
                onClick={() => setLogFilter("system")}
                className={cn(
                  "px-2 py-1 rounded transition-colors",
                  logFilter === "system" ? "bg-surface-container-lowest text-primary shadow-sm" : "text-outline hover:text-primary"
                )}
              >
                SYSTEM
              </button>
            </div>

            <div className="hidden sm:block border-l border-outline-variant h-5 mx-2 shrink-0" />

            {!failed && !allDone && (
              <span className="flex items-center gap-1.5 text-xs text-primary font-bold shrink-0">
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
                Running Node
              </span>
            )}
            {isComplete && (
              <span className="flex items-center gap-1.5 text-xs text-success-green font-bold shrink-0">
                <span className="w-1.5 h-1.5 rounded-full bg-success-green animate-pulse" />
                Ready · View Results ↓
              </span>
            )}
          </div>

          {/* Staggered cells notebooks content - Scrollable Container */}
          <div className="flex-1 overflow-y-auto w-full min-h-0 flex flex-col gap-3 pr-1.5">
            {STEPS.map((step, i) => {
              const status: StepStatus =
                failed && activeStep === i ? "error" :
                activeStep > i ? "done" :
                activeStep === i ? "active" : "waiting";
              return (
                <PipelineCell
                  key={step.key}
                  step={step}
                  index={i}
                  status={status}
                  cellNumber={i + 1}
                  elapsedAtDone={status === "done" ? stepElapsed[i] : undefined}
                  expanded={expandedCells.has(i)}
                  onToggle={() => toggleCell(i)}
                  logFilter={logFilter}
                />
              );
            })}

            {/* Pipeline complete - inline completion card */}
            <AnimatePresence>
              {isComplete && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: 10 }}
                  transition={{ duration: 0.35 }}
                  className="bg-surface-green-tint/25 border border-success-green/30 rounded-xl p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-surface-green-tint border border-success-green/30 flex items-center justify-center shrink-0">
                      <Icon name="check_circle" className="text-success-green" style={{ fontSize: "20px" }} />
                    </div>
                    <div className="text-left">
                      <p className="text-sm font-bold text-on-surface">All pipeline stages completed</p>
                      <p className="text-xs text-on-surface-variant mt-0.5">Champion model selected and metrics ready. Proceed when ready.</p>
                    </div>
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.03 }}
                    whileTap={{ scale: 0.97 }}
                    onClick={() => router.push(`/${runId}/result`)}
                    className="bg-success-green text-black text-sm font-bold px-6 py-2.5 rounded-lg flex items-center justify-center gap-2 shrink-0 transition-all hover:brightness-110 hover:shadow-[0_0_15px_rgba(95,208,127,0.5),0_0_30px_rgba(95,208,127,0.25)]"
                  >
                    View Results
                    <Icon name="arrow_forward" style={{ fontSize: "16px" }} />
                  </motion.button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
