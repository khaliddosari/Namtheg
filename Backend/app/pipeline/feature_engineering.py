from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype, is_timedelta64_dtype

from app import storage


HIGH_MISSING_THRESHOLD = 0.5
ONE_HOT_MAX_CARDINALITY = 10


def feature_engineer(run_id: str, target: str, extra_drops: list[dict] | None = None) -> dict:
    """Filter columns/rows and emit a ready-but-unencoded engineered dataset.

    Structural decisions live here (drop high-missing columns, drop id-like
    columns, drop rows with a missing target). Encoding is intentionally
    deferred to the training pipeline so each CV fold fits its own encoders
    on its own train portion - that closes the encoder-leakage source and
    lets the deployed Modal endpoint accept raw inputs (the Pipeline encodes
    them internally before predicting).

    `extra_drops` are the analyst agent's validated decisions
    ({"column", "reason"}), applied before the default rules.
    """
    df = storage.load_dataset(run_id)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")

    dropped: list[str] = []
    planned_encodings: list[str] = []

    # 0. Columns the analyst agent showed must not reach the model.
    for d in extra_drops or []:
        if d["column"] in df.columns and d["column"] != target:
            df = df.drop(columns=[d["column"]])
            dropped.append(f"{d['column']} ({d['reason'].replace('_', ' ')}, per analysis)")

    # 1. Drop columns with too many missing values (excluding target).
    for c in list(df.columns):
        if c == target:
            continue
        if df[c].isna().mean() > HIGH_MISSING_THRESHOLD:
            df = df.drop(columns=[c])
            dropped.append(f"{c} (>{int(HIGH_MISSING_THRESHOLD*100)}% missing)")

    # 2. Drop rows where target is missing (must precede ID-like check so n is correct).
    df = df.dropna(subset=[target]).reset_index(drop=True)
    n = len(df)

    # 3. Drop datetime columns. The encoders can't take raw timestamps, and
    # turning them into features safely needs a time-aware split (the audit
    # flags them). Custom transforms can't go in the pickled pipeline either:
    # the inference image couldn't load them.
    for c in list(df.columns):
        if c != target and (is_datetime64_any_dtype(df[c]) or is_timedelta64_dtype(df[c])):
            df = df.drop(columns=[c])
            dropped.append(f"{c} (date/time; not used as a feature yet)")

    # 4. Drop ID-like columns (>=95% unique values).
    # Numeric dtypes are intentionally exempt: continuous floats on small
    # datasets naturally hit ~100% uniqueness and would otherwise be wiped
    # out as if they were identifiers.
    for c in list(df.columns):
        if c == target:
            continue
        if is_numeric_dtype(df[c]):
            continue
        if df[c].nunique(dropna=True) >= 0.95 * n:
            df = df.drop(columns=[c])
            dropped.append(f"{c} (id-like)")

    # 5. Record the encoding plan. The actual encoders are fit by the GPU
    # trainer inside each CV fold - see app/training/core.py:build_preprocessor.
    for c in df.columns:
        if c == target or is_numeric_dtype(df[c]):
            continue
        cardinality = df[c].nunique(dropna=True)
        if cardinality <= ONE_HOT_MAX_CARDINALITY:
            planned_encodings.append(f"{c} (will be one-hot, {cardinality} levels)")
        else:
            planned_encodings.append(f"{c} (will be ordinal, {cardinality} levels)")

    storage.save_engineered(run_id, df)
    report = {
        "dropped_columns": dropped,
        "encoded_columns": planned_encodings,
        "final_feature_count": int(df.shape[1] - 1),
        "final_row_count": int(len(df)),
    }
    storage.write_json(run_id, "feature_engineering.json", report)
    return report
