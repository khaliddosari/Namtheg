"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { predict as predictApi, type ModelSchema, type PredictionResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Icon } from "./icon";

export function TryModelCard({
  runId,
  schema,
  deployed,
  variant = "full",
}: {
  runId: string;
  schema: ModelSchema | null;
  deployed: boolean;
  /** "full" shows the Form/JSON toggle. "simple" hides developer affordances for shared public links. */
  variant?: "full" | "simple";
}) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [mode, setMode] = useState<"form" | "json">("form");
  const [jsonText, setJsonText] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!schema) return;
    const initial: Record<string, string> = {};
    for (const c of schema.feature_cols) {
      const v = schema.sample[c];
      initial[c] = v === null || v === undefined ? "" : String(v);
    }
    setValues(initial);
    setJsonText(JSON.stringify({ features: schema.sample }, null, 2));
  }, [schema]);

  const onChange = (col: string, v: string) => {
    setValues((prev) => ({ ...prev, [col]: v }));
  };

  const resetToSample = () => {
    if (!schema) return;
    const initial: Record<string, string> = {};
    for (const c of schema.feature_cols) {
      const v = schema.sample[c];
      initial[c] = v === null || v === undefined ? "" : String(v);
    }
    setValues(initial);
    setError(null);
    setResult(null);
  };

  const runPredict = async () => {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      let payload: Parameters<typeof predictApi>[1];
      if (mode === "json") {
        try {
          payload = JSON.parse(jsonText);
        } catch (e) {
          throw new Error("Invalid JSON: " + (e instanceof Error ? e.message : String(e)));
        }
      } else {
        const features: Record<string, number | string | null> = {};
        for (const c of schema?.feature_cols ?? []) {
          const raw = values[c];
          if (raw === "" || raw === undefined) {
            features[c] = null;
          } else if (!isNaN(Number(raw))) {
            features[c] = Number(raw);
          } else {
            features[c] = raw;
          }
        }
        payload = { features };
      }
      const res = await predictApi(runId, payload);
      setResult(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  if (!deployed) {
    return (
      <motion.section
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="glass p-6 text-left opacity-60"
      >
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-surface-container flex items-center justify-center shrink-0">
            <Icon name="science" className="text-on-surface-variant" style={{ fontSize: "24px" }} />
          </div>
          <div>
            <h3 className="text-headline-md font-bold text-on-background">Try the Model</h3>
            <p className="text-xs font-medium text-on-surface-variant mt-0.5">
              {variant === "simple"
                ? "This model isn't live right now. Check back in a moment."
                : "Deploy the model to Modal first, then come back to send live prediction requests from here."}
            </p>
          </div>
        </div>
      </motion.section>
    );
  }

  const isClassification = schema?.problem_type === "classification";

  // Forecasts and images have no column form; they're served by the same
  // endpoint with a different body (see the model package's README).
  if (schema?.task === "forecasting" || schema?.task === "image_classification") {
    return (
      <motion.section
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.05 }}
        className="glass p-6 text-left"
      >
        <h3 className="text-headline-md font-bold text-on-background">Try the Model</h3>
        <p className="text-xs font-medium text-on-surface-variant mt-1">
          {schema.task === "forecasting"
            ? 'This forecaster is live. POST {} to the predict endpoint for the next steps after the training data, or {"history": [{"timestamp", "value"}]} to forecast from your own history.'
            : 'This image model is live. POST {"images": ["<base64 image>"]} to the predict endpoint.'}
          {" "}The downloadable model package runs the same predictions locally.
        </p>
      </motion.section>
    );
  }

  return (
    <motion.section
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.05 }}
      className="glass p-6 text-left"
    >
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div className="flex items-center gap-4 min-w-0">
          <div className="w-12 h-12 rounded-xl bg-surface-purple-tint flex items-center justify-center shrink-0">
            <Icon name="science" className="text-primary" style={{ fontSize: "24px" }} />
          </div>
          <div className="min-w-0">
            <h3 className="text-headline-md font-bold text-on-background">Try the Model</h3>
            <p className="text-xs font-medium text-on-surface-variant mt-0.5">
              {variant === "simple"
                ? "Fill in the values below and click Predict to see what the model says."
                : "Send a live request to the deployed endpoint. Pre-filled with a sample row - tweak any value to see how it changes."}
            </p>
          </div>
        </div>

        {schema && variant === "full" && (
          <div className="flex items-center gap-1 bg-surface-container rounded-lg p-1 border border-outline-variant">
            <button
              onClick={() => setMode("form")}
              className={cn(
                "px-3 py-1.5 text-xs font-bold rounded transition-colors cursor-pointer",
                mode === "form" ? "bg-primary text-on-primary" : "text-on-surface-variant hover:bg-surface-container-high",
              )}
            >
              Form
            </button>
            <button
              onClick={() => setMode("json")}
              className={cn(
                "px-3 py-1.5 text-xs font-bold rounded transition-colors cursor-pointer",
                mode === "json" ? "bg-primary text-on-primary" : "text-on-surface-variant hover:bg-surface-container-high",
              )}
            >
              JSON
            </button>
          </div>
        )}
      </div>

      {!schema ? (
        <div className="mt-5 flex items-center gap-2 text-xs text-on-surface-variant">
          <Icon name="sync" className="animate-spin" style={{ fontSize: "16px" }} />
          Loading model schema…
        </div>
      ) : (
        <>
          {mode === "form" ? (
            <div className="mt-5 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 max-h-80 overflow-y-auto pr-2">
              {schema.feature_cols.map((c) => (
                <div key={c} className="flex flex-col">
                  <label className="text-[10px] font-bold text-on-surface-variant uppercase tracking-wider font-mono truncate" title={c}>
                    {c}
                  </label>
                  <input
                    type="text"
                    value={values[c] ?? ""}
                    onChange={(e) => onChange(c, e.target.value)}
                    className="mt-1 px-2.5 py-1.5 bg-surface-container rounded-md border border-outline-variant text-sm font-mono focus:outline-none focus:border-primary"
                  />
                </div>
              ))}
            </div>
          ) : (
            <div className="mt-5">
              <textarea
                value={jsonText}
                onChange={(e) => setJsonText(e.target.value)}
                rows={10}
                className="w-full px-3 py-2 bg-surface-container rounded-md border border-outline-variant text-xs font-mono focus:outline-none focus:border-primary"
                spellCheck={false}
              />
              <p className="text-[10px] text-on-surface-variant mt-1 font-mono">
                Use <code className="bg-surface-container px-1 py-0.5 rounded">{`{"features": {...}}`}</code> for a single row, or <code className="bg-surface-container px-1 py-0.5 rounded">{`{"rows": [[...], ...]}`}</code> for a batch.
              </p>
            </div>
          )}

          <div className="mt-5 flex flex-col gap-4">
            {mode === "form" && (
              <button
                onClick={resetToSample}
                className="self-center text-sm font-bold text-primary hover:bg-surface-purple-tint px-4 py-2.5 rounded-lg transition-colors flex items-center gap-1.5 cursor-pointer"
              >
                <Icon name="restart_alt" style={{ fontSize: "18px" }} />
                Reset to sample
              </button>
            )}
            <button
              onClick={runPredict}
              disabled={busy}
              className={cn(
                "w-full px-5 py-2.5 rounded-lg text-sm font-bold flex items-center justify-center gap-2 transition-all cursor-pointer",
                busy ? "bg-surface-container text-on-surface-variant cursor-not-allowed" : "btn-primary hover:opacity-95",
              )}
            >
              {busy ? (
                <>
                  <Icon name="sync" className="animate-spin" style={{ fontSize: "18px" }} />
                  Predicting…
                </>
              ) : (
                <>
                  <Icon name="send" style={{ fontSize: "18px" }} />
                  Predict
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="mt-4 p-3 bg-error/5 border border-error/30 rounded-lg flex items-start gap-2">
              <Icon name="error" className="text-error shrink-0" style={{ fontSize: "16px" }} />
              <p className="text-xs text-error font-mono break-words">{error}</p>
            </div>
          )}

          {result && !result.error && (
            <div className="mt-5 bg-surface-container rounded-lg border border-outline-variant p-4">
              <h4 className="text-xs font-bold uppercase tracking-wider text-on-surface-variant font-mono mb-3 text-center">Prediction Result</h4>
              {isClassification ? (
                <div className="space-y-3">
                  {(result.predicted_labels ?? result.predictions ?? []).map((label, i) => {
                    const probs = result.probabilities?.[i];
                    const top = probs ? Math.max(...probs) : null;
                    return (
                      <div key={i} className="flex flex-col gap-2">
                        <div className="flex items-baseline gap-3 flex-wrap">
                          <span className="text-xs font-bold text-on-surface-variant font-mono">Predicted:</span>
                          <span className="text-2xl font-black text-primary font-mono">{String(label)}</span>
                          {top !== null && (
                            <span className="text-xs font-bold text-success-green font-mono">{(top * 100).toFixed(1)}% confidence</span>
                          )}
                        </div>
                        {probs && result.class_labels && (
                          <div className="space-y-1.5">
                            {result.class_labels.map((cls, j) => (
                              <div key={cls}>
                                <div className="flex justify-between text-[10px] font-mono font-bold mb-0.5">
                                  <span className="text-on-surface">{cls}</span>
                                  <span className="text-on-surface-variant">{(probs[j] * 100).toFixed(2)}%</span>
                                </div>
                                <div className="w-full bg-surface-variant h-2 rounded-full overflow-hidden">
                                  <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${probs[j] * 100}%` }}
                                    transition={{ duration: 0.5 }}
                                    className={cn("h-full rounded-full", j === probs.indexOf(top!) ? "bg-primary" : "bg-outline")}
                                  />
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="space-y-2 text-center">
                  {(result.predictions ?? []).map((p, i) => (
                    <div key={i} className="text-3xl font-black text-primary font-mono">
                      {typeof p === "number" ? p.toFixed(4) : String(p)}
                    </div>
                  ))}
                </div>
              )}
              {variant === "full" && (
                <details className="mt-4">
                  <summary className="text-[10px] font-bold text-on-surface-variant cursor-pointer hover:text-primary font-mono uppercase tracking-wider">
                    Raw response
                  </summary>
                  <pre className="mt-2 bg-surface px-3 py-2 rounded text-[11px] font-mono whitespace-pre-wrap break-all max-h-48 overflow-auto border border-outline-variant">
                    {JSON.stringify(result, null, 2)}
                  </pre>
                </details>
              )}
            </div>
          )}

          {result?.error && (
            <div className="mt-4 p-3 bg-error/5 border border-error/30 rounded-lg flex items-start gap-2">
              <Icon name="error" className="text-error shrink-0" style={{ fontSize: "16px" }} />
              <p className="text-xs text-error font-mono break-words">{result.error}</p>
            </div>
          )}
        </>
      )}
    </motion.section>
  );
}
