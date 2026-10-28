import pandas as pd
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype, is_timedelta64_dtype

from app import storage
from app.training.runtime import is_code, new_column_sources, prepare


HIGH_MISSING_THRESHOLD = 0.5
ONE_HOT_MAX_CARDINALITY = 10


def feature_engineer(run_id: str, target: str, extra_drops: list[dict] | None = None,
                     df: pd.DataFrame | None = None, cleaning: dict | None = None) -> dict:
    """Filter columns/rows and emit a ready-but-unencoded engineered dataset.

    Structural decisions live here (drop high-missing columns, drop id-like
    columns, drop rows with a missing target). Encoding is intentionally
    deferred to the training pipeline so each CV fold fits its own encoders
    on its own train portion - that closes the encoder-leakage source and
    lets the deployed Modal endpoint accept raw inputs (the Pipeline encodes
    them internally before predicting).

    `extra_drops` are the analyst agent's validated decisions ({"column",
    "reason"}), applied before the default rules. `cleaning` is its validated
    cleaning (analyst.cleaning_spec), with the new columns the user asked for:
    label merges change the target; the rest
    goes through runtime.prepare, the same function the model runs on every
    input at prediction time, so the cleaning that survives the rules below is
    recorded for the model bundle. `df` is the run's working set
    (app/pipeline/sampling.py) when the caller already holds it.
    """
    df = (storage.load_dataset(run_id) if df is None else df).copy()
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    cleaning = cleaning or {}

    dropped: list[str] = []
    planned_encodings: list[str] = []
    applied: list[str] = []

    # 0. Columns the analyst agent showed must not reach the model.
    for d in extra_drops or []:
        if d["column"] in df.columns and d["column"] != target:
            df = df.drop(columns=[d["column"]])
            dropped.append(f"{d['column']} ({d['reason'].replace('_', ' ')}, per analysis)")

    # 1. The analyst's cleaning, before the rules below judge the columns: a
    # column that is mostly "-999" is mostly missing.
    for merge in cleaning.get("label_merges") or []:
        values = list(df[target].dropna().unique())
        into = next((v for v in values if str(v).strip() == merge["into"]), None)
        variants = [v for v in values if str(v).strip() in merge["labels"] and v != into]
        if into is None or not variants:
            continue
        n = int(df[target].isin(variants).sum())
        df[target] = df[target].replace({v: into for v in variants})
        applied.append(f"Merged the target labels {', '.join(map(str, variants))} into {into} ({n} rows).")
    feature_cleaning = {
        "missing_codes": {c: v for c, v in (cleaning.get("missing_codes") or {}).items() if c in df and c != target},
        "new_columns": [s for s in cleaning.get("new_columns") or []
                        if s["name"] not in df and all(c in df and c != target for c in new_column_sources(s))],
        "categorical": [c for c in cleaning.get("categorical") or [] if c in df and c != target],
        "date_parts": {c: p for c, p in (cleaning.get("date_parts") or {}).items() if c in df and c != target},
    }
    for col, codes in feature_cleaning["missing_codes"].items():
        n = int(is_code(df[col], codes).sum())
        applied.append(f"Treated {', '.join(map(str, codes))} in {col} as missing ({n} values).")
    for spec in feature_cleaning["new_columns"]:
        how = spec.get("formula") or f"{spec['bins']['column']} in ranges from {', '.join(f'{e:g}' for e in spec['bins']['edges'])}"
        applied.append(f"Added the column {spec['name']} you asked for: {how}.")
    for col in feature_cleaning["categorical"]:
        applied.append(f"Treated {col} as categories, not quantities ({df[col].nunique(dropna=True)} codes).")
    for col, parts in feature_cleaning["date_parts"].items():
        applied.append(f"Turned {col} into {', '.join(p.replace('_', ' ') for p in parts)}.")
    if any(feature_cleaning.values()):
        df = prepare(df.drop(columns=[target]), feature_cleaning).assign(**{target: df[target]})

    # 2. Drop columns with too many missing values (excluding target).
    for c in list(df.columns):
        if c == target:
            continue
        if df[c].isna().mean() > HIGH_MISSING_THRESHOLD:
            df = df.drop(columns=[c])
            dropped.append(f"{c} (>{int(HIGH_MISSING_THRESHOLD*100)}% missing)")

    # 3. Drop rows where target is missing (must precede ID-like check so n is correct).
    df = df.dropna(subset=[target]).reset_index(drop=True)
    n = len(df)

    # 4. Drop the remaining datetime columns. The encoders can't take raw
    # timestamps; a date only becomes features when the analyst asked for its
    # calendar parts (step 1).
    for c in list(df.columns):
        if c != target and (is_datetime64_any_dtype(df[c]) or is_timedelta64_dtype(df[c])):
            df = df.drop(columns=[c])
            dropped.append(f"{c} (date/time; not turned into features)")

    # 5. Drop ID-like columns (>=95% unique values). Numeric dtypes are exempt:
    # continuous floats on small datasets naturally hit ~100% uniqueness.
    for c in list(df.columns):
        if c == target:
            continue
        if is_numeric_dtype(df[c]):
            continue
        if df[c].nunique(dropna=True) >= 0.95 * n:
            df = df.drop(columns=[c])
            dropped.append(f"{c} (id-like)")

    # 6. Record the encoding plan. The actual encoders are fit by the GPU
    # trainer inside each CV fold - see app/training/tabular.py:build_preprocessor.
    for c in df.columns:
        if c == target or is_numeric_dtype(df[c]):
            continue
        cardinality = df[c].nunique(dropna=True)
        if cardinality <= ONE_HOT_MAX_CARDINALITY:
            planned_encodings.append(f"{c} (will be one-hot, {cardinality} levels)")
        else:
            planned_encodings.append(f"{c} (will be ordinal, {cardinality} levels)")

    # The raw columns a prediction needs: a date part comes from its date
    # column, and a new column from the columns it is built from.
    model_cols = [c for c in df.columns if c != target]
    sources = {f"{c}__{p}": [c] for c, parts in feature_cleaning["date_parts"].items() for p in parts}
    sources.update({s["name"]: new_column_sources(s) for s in feature_cleaning["new_columns"]})
    input_cols = list(dict.fromkeys(src for c in model_cols for src in sources.get(c, [c])))
    kept = set(input_cols)
    kept_cleaning = {
        "missing_codes": {c: v for c, v in feature_cleaning["missing_codes"].items() if c in kept},
        "new_columns": [s for s in feature_cleaning["new_columns"] if s["name"] in model_cols],
        "categorical": [c for c in feature_cleaning["categorical"] if c in kept],
        "date_parts": {c: [p for p in parts if f"{c}__{p}" in model_cols]
                       for c, parts in feature_cleaning["date_parts"].items() if c in kept},
    }

    storage.save_engineered(run_id, df)
    report = {
        "dropped_columns": dropped,
        "encoded_columns": planned_encodings,
        "cleaning_applied": applied,
        "input_columns": input_cols,
        "cleaning": kept_cleaning if any(kept_cleaning.values()) else None,
        "final_feature_count": len(model_cols),
        "final_row_count": int(len(df)),
    }
    storage.write_json(run_id, "feature_engineering.json", report)
    return report
