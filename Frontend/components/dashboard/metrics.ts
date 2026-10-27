// Selection metrics the trainer can report (imbalanced targets use macro F1).
// percent: shown as a percentage; otherwise a plain number. lowerIsBetter flips
// ranking and wording.
const METRICS: Record<string, { short: string; long: string; percent: boolean; lowerIsBetter?: boolean }> = {
  accuracy: { short: "Accuracy", long: "Cross-validation accuracy", percent: true },
  f1_macro: { short: "Macro F1", long: "Cross-validation macro F1", percent: true },
  r2: { short: "R²", long: "R² regression score", percent: true },
};

export const metricInfo = (metric: string) => METRICS[metric] ?? { short: metric, long: metric, percent: false };

export const formatScore = (value: number, metric: string, digits = 2) =>
  metricInfo(metric).percent ? `${(value * 100).toFixed(digits)}%` : value.toFixed(4);
