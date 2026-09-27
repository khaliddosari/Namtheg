from pandas.api.types import is_numeric_dtype

from app import storage


def run_eda(run_id: str, target: str) -> dict:
    df = storage.load_dataset(run_id)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    numeric_summary = df[numeric_cols].describe().round(4).to_dict() if numeric_cols else {}

    if is_numeric_dtype(df[target]):
        target_distribution = {
            "mean": float(df[target].mean()),
            "std": float(df[target].std()),
            "min": float(df[target].min()),
            "max": float(df[target].max()),
        }
        corr = {}
        for c in numeric_cols:
            if c == target:
                continue
            try:
                corr[c] = round(float(df[c].corr(df[target])), 4)
            except Exception:
                continue
        correlations = dict(
            sorted(corr.items(), key=lambda x: abs(x[1]) if x[1] == x[1] else 0, reverse=True)[:10]
        )
    else:
        counts = df[target].value_counts(dropna=True).head(20).to_dict()
        target_distribution = {str(k): int(v) for k, v in counts.items()}
        correlations = {}

    report = {
        "target": target,
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "numeric_summary": numeric_summary,
        "target_distribution": target_distribution,
        "correlations_with_target": correlations,
        "missing_per_column": {c: int(df[c].isna().sum()) for c in df.columns},
    }
    storage.write_json(run_id, "eda.json", report)
    return report
