"use client";

import type { RunResult } from "@/lib/api";
import { cn } from "@/lib/utils";
import { formatArabicBrand } from "@/components/brand";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";

const BASIS = { data: "From the data", name: "From the name", both: "Name and data" } as const;
const SEVERITY = {
  critical: { label: "Critical", className: "text-red-300" },
  warning: { label: "Warning", className: "text-amber-300" },
  info: { label: "Note", className: "text-muted-foreground" },
} as const;

/** Whether a result has anything for the analyst dialog to show. */
export function hasAnalysis(result: RunResult): boolean {
  const a = result.extra?.analysis;
  if (!a) return false;
  return !!(
    a.data_dictionary ||
    a.findings?.length ||
    a.open_questions?.length ||
    a.drop_columns?.length ||
    result.extra?.cleaning_applied?.length ||
    a.notes?.length
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="space-y-2">
      <h3 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">{title}</h3>
      {children}
    </section>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-[6.5rem_minmax(0,1fr)] gap-3">
      <dt className="truncate text-muted-foreground" title={label}>
        {label}
      </dt>
      <dd className="min-w-0 break-words">{children}</dd>
    </div>
  );
}

/**
 * What the analyst agent understood before training: what a row and the target
 * are, what the important columns mean (and whether that rests on the data or
 * only on a column's name), the cleaning applied, findings, and the questions
 * only the data owner can answer.
 */
export function AnalysisDialog({
  result,
  open,
  onOpenChange,
}: {
  result: RunResult;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const extra = result.extra ?? {};
  const analysis = extra.analysis ?? {};
  const dictionary = analysis.data_dictionary;
  const sample = extra.sample;
  const cleaning = extra.cleaning_applied ?? [];
  const drops = analysis.drop_columns ?? [];
  const findings = analysis.findings ?? [];
  const questions = analysis.open_questions ?? [];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>What the analyst found</DialogTitle>
          <DialogDescription>
            {analysis.summary ? formatArabicBrand(analysis.summary) : "How the analyst agent read this data before training."}
          </DialogDescription>
        </DialogHeader>
        <div
          role="region"
          aria-label="Analysis details"
          tabIndex={0}
          className="-mx-1 max-h-[60vh] space-y-5 overflow-y-auto rounded-md px-1 text-sm outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
        >
          {(dictionary || sample?.sampled || analysis.prediction_time !== undefined) && (
            <Section title="The data">
              <dl className="space-y-1.5">
                {dictionary?.row_meaning && <Row label="One row">{dictionary.row_meaning}</Row>}
                {dictionary?.target_meaning && <Row label={result.target ?? "Target"}>{dictionary.target_meaning}</Row>}
                {analysis.prediction_time !== undefined && (
                  <Row label="Predicting">
                    {analysis.prediction_time ?? "Not stated, so the model could use anything recorded before the outcome."}
                  </Row>
                )}
                {sample?.sampled && (
                  <Row label="Rows">
                    Trained on a random sample of {sample.rows_used.toLocaleString()} of {sample.rows_total.toLocaleString()} rows,
                    the same rows the analyst studied.
                  </Row>
                )}
              </dl>
            </Section>
          )}

          {!!dictionary?.columns.length && (
            <Section title="Columns">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead className="h-8 px-2 text-xs text-muted-foreground">Column</TableHead>
                    <TableHead className="h-8 px-2 text-xs text-muted-foreground">Meaning</TableHead>
                    <TableHead className="h-8 px-2 text-xs text-muted-foreground">Basis</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {dictionary.columns.map((c) => (
                    <TableRow key={c.column}>
                      <TableCell className="px-2 py-1.5 align-top font-mono text-xs">{c.column}</TableCell>
                      <TableCell className="px-2 py-1.5 align-top whitespace-normal">{c.meaning}</TableCell>
                      <TableCell className="px-2 py-1.5 align-top text-xs text-muted-foreground">{BASIS[c.basis] ?? c.basis}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </Section>
          )}

          {!!dictionary?.relationships.length && (
            <Section title="How the columns relate">
              <ul className="space-y-1.5">
                {dictionary.relationships.map((r, i) => (
                  <li key={i}>
                    {r.finding} <span className="text-muted-foreground">{r.evidence}</span>
                  </li>
                ))}
              </ul>
            </Section>
          )}

          {(cleaning.length > 0 || drops.length > 0) && (
            <Section title="Cleaning before training">
              <ul className="list-disc space-y-1.5 ps-5 marker:text-muted-foreground">
                {cleaning.map((c, i) => (
                  <li key={`c${i}`}>{c}</li>
                ))}
                {drops.map((d) => (
                  <li key={`d${d.column}`}>
                    Dropped <span className="font-mono text-xs">{d.column}</span> ({d.reason.replaceAll("_", " ")}).{" "}
                    <span className="text-muted-foreground">{d.evidence}</span>
                  </li>
                ))}
              </ul>
            </Section>
          )}

          {findings.length > 0 && (
            <Section title="Findings">
              <ul className="space-y-1.5">
                {findings.map((f, i) => {
                  const severity = SEVERITY[f.severity] ?? SEVERITY.info;
                  return (
                    <li key={i}>
                      <span className={cn("font-medium", severity.className)}>{severity.label}:</span> {f.finding}{" "}
                      <span className="text-muted-foreground">{f.evidence}</span>
                    </li>
                  );
                })}
              </ul>
            </Section>
          )}

          {questions.length > 0 && (
            <Section title="Open questions for you">
              <ul className="list-disc space-y-1.5 ps-5 marker:text-muted-foreground">
                {questions.map((q, i) => (
                  <li key={i}>{q}</li>
                ))}
              </ul>
            </Section>
          )}

          {analysis.status !== "completed" && !!analysis.notes?.length && (
            <Section title="Notes">
              <ul className="list-disc space-y-1.5 ps-5 marker:text-muted-foreground">
                {analysis.notes.map((n, i) => (
                  <li key={i}>{n}</li>
                ))}
              </ul>
            </Section>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
