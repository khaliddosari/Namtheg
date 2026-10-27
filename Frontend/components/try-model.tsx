"use client";

import { useEffect, useState } from "react";
import { RotateCcw, Send } from "lucide-react";
import { predict as predictApi, type ModelSchema, type PredictionResponse } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Spinner } from "@/components/ui/spinner";
import { Field, FieldLabel } from "@/components/ui/field";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Meter } from "@/components/panel";
import { Segmented } from "@/components/segmented";

const sampleValues = (schema: ModelSchema) =>
  Object.fromEntries(
    schema.feature_cols.map((c) => {
      const v = schema.sample[c];
      return [c, v === null || v === undefined ? "" : String(v)];
    })
  ) as Record<string, string>;

/**
 * Live predictions against the deployed endpoint, pre-filled with a sample row.
 * "full" adds the JSON body editor and the raw response for developers; "simple"
 * is the form alone, for the public share page. "split" puts the answer beside
 * the form on a locked dashboard page; "stack" puts it underneath.
 */
export function TryModel({
  runId,
  schema,
  variant = "full",
  layout = "stack",
}: {
  runId: string;
  schema: ModelSchema;
  variant?: "full" | "simple";
  layout?: "split" | "stack";
}) {
  const [values, setValues] = useState<Record<string, string>>(() => sampleValues(schema));
  const [mode, setMode] = useState<"form" | "json">("form");
  const [jsonText, setJsonText] = useState(() => JSON.stringify({ features: schema.sample }, null, 2));
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setValues(sampleValues(schema));
    setJsonText(JSON.stringify({ features: schema.sample }, null, 2));
  }, [schema]);

  const reset = () => {
    setValues(sampleValues(schema));
    setError(null);
    setResult(null);
  };

  const run = async () => {
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
        for (const c of schema.feature_cols) {
          const raw = values[c];
          features[c] = raw === "" || raw === undefined ? null : !isNaN(Number(raw)) ? Number(raw) : raw;
        }
        payload = { features };
      }
      setResult(await predictApi(runId, payload));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const isClassification = schema.problem_type === "classification";

  return (
    <div className={cn("flex min-h-0 flex-1 flex-col gap-4", layout === "split" && "lock:grid lock:grid-cols-[minmax(0,1fr)_minmax(16rem,22rem)]")}>
      <div className="flex min-h-0 flex-col gap-3">
        <div className="flex flex-wrap items-center gap-2">
          {variant === "full" && (
            <Segmented
              label="Request format"
              value={mode}
              onChange={setMode}
              options={[
                { value: "form", label: "Form" },
                { value: "json", label: "JSON" },
              ]}
            />
          )}
          {mode === "form" && (
            <Button variant="ghost" size="sm" onClick={reset} className="ms-auto h-9 lg:h-7">
              <RotateCcw data-icon="inline-start" />
              Reset to sample
            </Button>
          )}
        </div>

        {mode === "form" ? (
          <div className="@container -me-1 max-h-[26rem] min-h-0 flex-1 overflow-y-auto pe-1 lock:max-h-none">
            <div className="grid gap-3 @md:grid-cols-2 @2xl:grid-cols-3">
              {schema.feature_cols.map((c) => (
                <Field key={c} className="gap-1.5">
                  <FieldLabel htmlFor={`feature-${c}`} className="block w-full truncate font-mono text-xs font-normal text-muted-foreground" title={c}>
                    {c}
                  </FieldLabel>
                  <Input
                    id={`feature-${c}`}
                    value={values[c] ?? ""}
                    onChange={(e) => setValues((prev) => ({ ...prev, [c]: e.target.value }))}
                    dir="ltr"
                    className="h-10 font-mono lg:h-8"
                  />
                </Field>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex min-h-0 flex-1 flex-col gap-1.5">
            <Textarea
              aria-label="Request body"
              value={jsonText}
              onChange={(e) => setJsonText(e.target.value)}
              spellCheck={false}
              dir="ltr"
              className="min-h-48 flex-1 resize-none font-mono text-xs"
            />
            <p className="text-xs text-muted-foreground">
              <code className="font-mono">{'{"features": {...}}'}</code> for one row, <code className="font-mono">{'{"rows": [[...], ...]}'}</code> for a
              batch.
            </p>
          </div>
        )}

        <Button onClick={run} disabled={busy} className="h-10 lg:h-8">
          {busy ? <Spinner data-icon="inline-start" /> : <Send data-icon="inline-start" />}
          {busy ? "Predicting..." : "Predict"}
        </Button>
      </div>

      <div className="flex min-h-0 flex-col gap-3 overflow-y-auto rounded-xl bg-background/40 p-3 ring-1 ring-border" aria-live="polite">
        <h3 className="text-xs font-medium text-muted-foreground">Prediction</h3>
        {error || result?.error ? (
          <p role="alert" className="text-xs break-words text-red-300">
            {error ?? result?.error}
          </p>
        ) : result ? (
          isClassification ? (
            (result.predicted_labels ?? result.predictions ?? []).map((raw, i) => {
              const label = raw ?? result.predictions?.[i] ?? "-";
              const probs = result.probabilities?.[i];
              const top = probs ? Math.max(...probs) : null;
              return (
                <div key={i} className="flex flex-col gap-2">
                  <div className="flex flex-wrap items-baseline gap-x-3">
                    <span className="font-mono text-2xl font-bold break-all text-primary">{String(label)}</span>
                    {top !== null && <span className="font-mono text-xs text-muted-foreground tabular-nums">{(top * 100).toFixed(1)}% confidence</span>}
                  </div>
                  {probs && result.class_labels && (
                    <ul className="flex flex-col gap-1.5">
                      {result.class_labels.map((cls, j) => (
                        <li key={cls}>
                          <div className="mb-0.5 flex justify-between gap-2 font-mono text-xs">
                            <span className="truncate">{cls}</span>
                            <span className="text-muted-foreground tabular-nums">{(probs[j] * 100).toFixed(2)}%</span>
                          </div>
                          <Meter value={probs[j] * 100} barClassName={probs[j] === top ? "bg-chart-1" : "bg-chart-4"} />
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              );
            })
          ) : (
            <div className="flex flex-col gap-1">
              {(result.predictions ?? []).map((p, i) => (
                <p key={i} className="font-mono text-3xl font-bold break-all text-primary tabular-nums">
                  {typeof p === "number" ? p.toFixed(4) : String(p)}
                </p>
              ))}
            </div>
          )
        ) : (
          <p className="text-sm text-muted-foreground">Send a request to see what the model says.</p>
        )}

        {variant === "full" && result && !result.error && (
          <Collapsible>
            <CollapsibleTrigger render={<Button variant="link" size="sm" className="h-auto px-0" />}>Raw response</CollapsibleTrigger>
            <CollapsibleContent>
              <pre className="mt-2 max-h-48 overflow-auto rounded-lg bg-background/60 px-3 py-2 font-mono text-[0.6875rem] break-all whitespace-pre-wrap ring-1 ring-border" dir="ltr">
                {JSON.stringify(result, null, 2)}
              </pre>
            </CollapsibleContent>
          </Collapsible>
        )}
      </div>
    </div>
  );
}
