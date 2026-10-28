"""Deterministic data-quality and leakage checks, run before any model sees the data.

The audit only flags; it never changes the data. Its findings are the starting
evidence for the analyst agent and are surfaced as-is in the final result, so
every check reports the numbers behind it.
"""
import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_datetime64_any_dtype, is_numeric_dtype

from app import storage
from app.pipeline import imbalance

# Leakage checks run on a sample so wide/long datasets stay fast.
LEAKAGE_SAMPLE_ROWS = 20_000
LEAKAGE_MAX_FEATURES = 200
LEAKAGE_BINS = 16
LEAKAGE_MAX_CATEGORIES = 50
LEAKAGE_FOLDS = 3
# A single feature explaining the target this well is almost always leakage
# (the feature is derived from the target, or recorded after the outcome).
LEAKAGE_SCORE_THRESHOLD = 0.98
# Columns can also leak together: an arrival delay is the departure delay plus
# the taxi and flight times minus the scheduled time, so those columns rebuild it
# by plain addition even though none does alone. A linear formula of a few
# columns explaining this much of the target is adding up, not predicting (on a
# flight table the honest columns reach 0.9475, the leaking ones 0.984).
FORMULA_R2 = 0.97
FORMULA_MAX_COLUMNS = 6
FORMULA_MIN_COMPLETE = 0.9  # share of sampled rows a formula's columns must all fill
# A class target can be decided by a number (a late flag by the delay in minutes);
# that number is then checked for a formula instead.
PARENT_MIN_UNIQUE = 20
MIN_SAMPLES_PER_CLASS = 10
TEXT_PATTERN_SAMPLE = 500
TEXT_PATTERN_MIN_SHARE = 0.9
_NUMERIC_TEXT_JUNK = r"[,\s$€£%]"

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def _finding(check: str, severity: str, detail: str, column: str | None = None, **values) -> dict:
    return {"check": check, "severity": severity, "column": column, "detail": detail, **values}


def _pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 2) if whole else 0.0


def _text_sample(s: pd.Series) -> pd.Series:
    values = s.dropna()
    return values.sample(min(len(values), TEXT_PATTERN_SAMPLE), random_state=0).astype(str)


def _looks_numeric_text(s: pd.Series) -> float:
    sample = _text_sample(s)
    if sample.empty:
        return 0.0
    cleaned = sample.str.replace(_NUMERIC_TEXT_JUNK, "", regex=True)
    return float(pd.to_numeric(cleaned, errors="coerce").notna().mean())


def _looks_datetime_text(s: pd.Series) -> float:
    sample = _text_sample(s)
    # Bare numbers parse as dates too; they aren't evidence of a date column.
    sample = sample[pd.to_numeric(sample, errors="coerce").isna()]
    if sample.empty:
        return 0.0
    return float(pd.to_datetime(sample, errors="coerce", format="mixed").notna().mean())


def _bin(x: pd.Series) -> pd.Series:
    """Discretise one feature: quantile bins for numbers, top categories for text.
    Missing values get their own bin, since missingness can itself leak."""
    if is_numeric_dtype(x) and not is_bool_dtype(x) and x.nunique() > LEAKAGE_BINS:
        binned = pd.qcut(x, LEAKAGE_BINS, labels=False, duplicates="drop").astype("float")
    else:
        top = x.value_counts().index[:LEAKAGE_MAX_CATEGORIES]
        binned = pd.Series(np.where(x.isin(top), x.astype(str), "__other__"), index=x.index)
    return binned.astype(str).where(x.notna(), "__missing__")


def _single_feature_score(x: pd.Series, y: pd.Series, problem_type: str) -> float | None:
    """How well one feature alone predicts the target, on held-out rows.

    Model-free: each fold builds a lookup table (bin -> majority class, or bin
    -> mean target) from the other folds and scores it on its own rows, so no
    model is trained here. Returns accuracy (classification) or, for
    regression, the larger of held-out R2 and squared rank correlation (binning
    caps R2 just below 1 even for an exact copy of the target; rank
    correlation catches any monotonic transform of it).
    """
    if len(x) < 30 or x.nunique(dropna=False) < 2:
        return None
    rank_r2 = 0.0
    if problem_type == "regression" and is_numeric_dtype(x) and not is_bool_dtype(x):
        rho = x.corr(y, method="spearman")
        rank_r2 = 0.0 if pd.isna(rho) else float(rho) ** 2
    bins = _bin(x).to_numpy()
    y = y.to_numpy()
    fold = np.random.default_rng(0).integers(0, LEAKAGE_FOLDS, size=len(y))
    preds = np.empty(len(y), dtype=object if problem_type == "classification" else float)
    for k in range(LEAKAGE_FOLDS):
        train, test = fold != k, fold == k
        if not train.any() or not test.any():
            return None
        frame = pd.DataFrame({"bin": bins[train], "y": y[train]})
        if problem_type == "classification":
            table = pd.crosstab(frame["bin"], frame["y"]).idxmax(axis=1)
            fallback = frame["y"].value_counts().index[0]
        else:
            table = frame.groupby("bin")["y"].mean()
            fallback = frame["y"].mean()
        preds[test] = pd.Series(bins[test]).map(table).fillna(fallback).to_numpy()
    if problem_type == "classification":
        return float((preds == y).mean())
    total = ((y - y.mean()) ** 2).sum()
    if not total:
        return None
    return max(float(1 - ((y - preds.astype(float)) ** 2).sum() / total), rank_r2)


def _held_out_r2(X: np.ndarray, y: np.ndarray) -> float:
    """R2 of a least-squares line, each fold scored by a fit on the other folds."""
    fold = np.random.default_rng(0).integers(0, LEAKAGE_FOLDS, size=len(y))
    sse = 0.0
    for k in range(LEAKAGE_FOLDS):
        train, test = fold != k, fold == k
        if not train.any() or not test.any():
            return 0.0
        coef, *_ = np.linalg.lstsq(np.column_stack([X[train], np.ones(train.sum())]), y[train], rcond=None)
        sse += float(((y[test] - np.column_stack([X[test], np.ones(test.sum())]) @ coef) ** 2).sum())
    total = float(((y - y.mean()) ** 2).sum())
    return 1 - sse / total if total else 0.0


def formula_leak(df: pd.DataFrame, target: str, columns: list[str]) -> dict | None:
    """The shortest linear formula of `columns` that rebuilds the numeric
    `target` with held-out R2 of at least FORMULA_R2, or None. Columns join one
    at a time, best first, up to FORMULA_MAX_COLUMNS; the search never stops
    early on a small gain, since parts of a sum can help only once their
    counterpart is in (time in the air, next to the scheduled time). Parts of
    a sum show up with weights near plus or minus one."""
    s = df[df[target].notna()] if target in df.columns else df.iloc[:0]
    if len(s) > LEAKAGE_SAMPLE_ROWS:
        s = s.sample(LEAKAGE_SAMPLE_ROWS, random_state=0)
    if len(s) < 30 or not is_numeric_dtype(s[target]) or is_bool_dtype(s[target]):
        return None
    names = [c for c in columns if c != target and c in s.columns and is_numeric_dtype(s[c])
             and not is_bool_dtype(s[c]) and s[c].nunique(dropna=True) > 1
             and s[c].notna().mean() >= FORMULA_MIN_COMPLETE][:LEAKAGE_MAX_FEATURES]
    if not names:
        return None
    X = s[names].to_numpy(dtype=float)
    y = s[target].to_numpy(dtype=float)
    chosen: list[int] = []
    for _ in range(min(FORMULA_MAX_COLUMNS, len(names))):
        best = None
        for j in range(len(names)):
            if j in chosen:
                continue
            cols = chosen + [j]
            rows = ~np.isnan(X[:, cols]).any(axis=1)
            if rows.mean() < FORMULA_MIN_COMPLETE:
                continue
            r2 = _held_out_r2(X[rows][:, cols], y[rows])
            if best is None or r2 > best[1]:
                best = (j, r2, rows)
        if best is None:
            return None
        chosen.append(best[0])
        if best[1] >= FORMULA_R2:
            rows = best[2]
            coef, *_ = np.linalg.lstsq(np.column_stack([X[rows][:, chosen], np.ones(rows.sum())]), y[rows], rcond=None)
            return {"target": target, "r2": round(best[1], 4), "rows": int(rows.sum()),
                    "weights": {names[j]: round(float(w), 3) for j, w in zip(chosen, coef[:-1])},
                    "constant": round(float(coef[-1]), 3)}
    return None


def formula_target(df: pd.DataFrame, target: str, problem_type: str, suspects: set[str]) -> str | None:
    """The number to check for a formula: a regression target itself; for a
    class target, the most finely measured numeric column that alone decides
    it (a leakage suspect), if there is one."""
    if problem_type != "classification":
        return target
    numbers = [c for c in suspects if c in df.columns and is_numeric_dtype(df[c]) and not is_bool_dtype(df[c])
               and df[c].nunique(dropna=True) >= PARENT_MIN_UNIQUE]
    return min(numbers, key=lambda c: (-df[c].nunique(dropna=True), c)) if numbers else None


def describe_formula(target: str, formula: dict) -> str:
    """The formula as text, and what it means for `target`."""
    terms = " ".join(f"{'-' if w < 0 else '+'} {abs(w):g} x {c}" for c, w in formula["weights"].items())
    sign = "-" if formula["constant"] < 0 else "+"
    text = (f"{formula['target']} ~ {terms.lstrip('+ ')} {sign} {abs(formula['constant']):g} "
            f"(held-out R2 {formula['r2']:.4f} on {formula['rows']} rows)")
    if formula["target"] != target:
        text = f"{target} is decided by {formula['target']}, and {text}"
    return text


def audit_dataset(run_id: str, target: str, problem_type: str, df: pd.DataFrame | None = None) -> dict:
    """`df`: the full table, when the caller already holds it. It is only read."""
    df = storage.load_dataset(run_id) if df is None else df
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    n = len(df)
    features = [c for c in df.columns if c != target]
    findings: list[dict] = []
    y = df[target]

    # -- target ------------------------------------------------------------
    target_missing = int(y.isna().sum())
    if target_missing:
        findings.append(_finding(
            "target_missing", "warning",
            f"{target_missing} rows ({_pct(target_missing, n)}%) have no target value and will be excluded from training.",
            target, count=target_missing, percent=_pct(target_missing, n),
        ))

    class_balance = None
    imbalance_info = imbalance.assess(y, problem_type)
    if problem_type == "classification":
        counts = y.value_counts(dropna=True)
        total = int(counts.sum())
        class_balance = {str(k): int(v) for k, v in counts.items()}
        if imbalance_info["detected"]:
            findings.append(_finding(
                "class_imbalance", "warning",
                f"Classes are imbalanced: the largest class has {imbalance_info['ratio']}x the rows of the "
                f"smallest ({counts.idxmin()!r}, {_pct(counts.min(), total)}% of rows). Training will use "
                "balanced class weights inside each training fold and pick models by macro F1, not accuracy.",
                target, ratio=imbalance_info["ratio"], minority_class=str(counts.idxmin()),
                minority_count=int(counts.min()), minority_percent=_pct(counts.min(), total),
            ))
        rare = counts[counts < MIN_SAMPLES_PER_CLASS]
        if len(rare):
            findings.append(_finding(
                "rare_classes", "warning",
                f"{len(rare)} class(es) have fewer than {MIN_SAMPLES_PER_CLASS} rows "
                f"({', '.join(f'{k!r}: {int(v)}' for k, v in rare.items())}); scores for them are unreliable.",
                target, classes={str(k): int(v) for k, v in rare.items()},
            ))
    elif is_numeric_dtype(y):
        skew = float(y.skew())
        if abs(skew) > 2:
            findings.append(_finding(
                "target_skew", "info",
                f"Target is heavily skewed (skewness {skew:.2f}); errors on the largest values will dominate RMSE.",
                target, skewness=round(skew, 2),
            ))

    # -- rows ----------------------------------------------------------------
    dup_rows = int(df.duplicated().sum())
    if dup_rows:
        findings.append(_finding(
            "duplicate_rows", "warning",
            f"{dup_rows} exact duplicate rows ({_pct(dup_rows, n)}%). Copies can land in both train and test "
            "folds and inflate scores.",
            count=dup_rows, percent=_pct(dup_rows, n),
        ))
    dup_features = int(df.duplicated(subset=features).sum()) - dup_rows if features else 0
    if dup_features > 0:
        findings.append(_finding(
            "conflicting_rows", "info",
            f"{dup_features} rows repeat another row's features but with a different target value.",
            count=dup_features,
        ))

    # -- columns -------------------------------------------------------------
    for c in features:
        s = df[c]
        missing = int(s.isna().sum())
        unique = int(s.nunique(dropna=True))
        if unique <= 1:
            findings.append(_finding("constant_column", "info",
                                     f"Column has {unique} distinct value(s) and carries no signal.", c))
            continue
        if missing / n > 0.5:
            findings.append(_finding("high_missing", "info",
                                     f"{_pct(missing, n)}% missing; dropped by the default pipeline.", c,
                                     percent=_pct(missing, n)))
        if is_numeric_dtype(s) and not is_bool_dtype(s):
            if s.dtype.kind in "iu" and unique == n and (s.is_monotonic_increasing or s.is_monotonic_decreasing):
                findings.append(_finding(
                    "row_identifier", "warning",
                    "Integer column with a unique, monotonic value per row: looks like a row number or ID. "
                    "IDs can leak ordering or collection effects into the model.", c,
                ))
            continue
        if is_datetime64_any_dtype(s):
            findings.append(_finding(
                "temporal_column", "warning",
                "Datetime column. If rows are ordered in time, a random train/test split lets the model see "
                "the future; a time-based split may be required.", c,
            ))
            continue
        if unique >= 0.95 * (n - missing):
            findings.append(_finding("identifier_like", "info",
                                     f"{unique} distinct values in {n - missing} rows: identifier-like text; "
                                     "dropped by the default pipeline.", c, unique=unique))
            continue
        numeric_share = _looks_numeric_text(s)
        if numeric_share >= TEXT_PATTERN_MIN_SHARE:
            findings.append(_finding(
                "numeric_stored_as_text", "warning",
                f"{_pct(numeric_share, 1)}% of sampled values are numbers once separators/currency/percent signs "
                "are removed, but the column is text, so it will be treated as categories.", c,
                percent=_pct(numeric_share, 1),
            ))
        elif _looks_datetime_text(s) >= TEXT_PATTERN_MIN_SHARE:
            findings.append(_finding(
                "temporal_column", "warning",
                "Text column whose values parse as dates. If rows are ordered in time, a random split lets the "
                "model see the future; the column is currently treated as categories.", c,
            ))

    # -- leakage suspects ------------------------------------------------------
    sample = df[df[target].notna()]
    if len(sample) > LEAKAGE_SAMPLE_ROWS:
        sample = sample.sample(LEAKAGE_SAMPLE_ROWS, random_state=0)
    candidates = [c for c in features if sample[c].nunique(dropna=True) > 1][:LEAKAGE_MAX_FEATURES]
    y_sample = sample[target]
    if problem_type == "classification":
        majority_rate = float(y_sample.value_counts(normalize=True).iloc[0]) if len(y_sample) else 1.0
    for c in candidates:
        score = _single_feature_score(sample[c], y_sample, problem_type)
        if score is None or score < LEAKAGE_SCORE_THRESHOLD:
            continue
        if problem_type == "classification" and majority_rate >= LEAKAGE_SCORE_THRESHOLD:
            continue  # target is nearly constant; high accuracy is not evidence of leakage
        metric = "held-out accuracy" if problem_type == "classification" else "held-out R2 / squared rank correlation"
        findings.append(_finding(
            "leakage_suspect", "critical",
            f"This column alone predicts the target almost perfectly ({metric} {score:.4f}). Verify it is known "
            "before the outcome and is not derived from the target; otherwise the model's score is not real.",
            c, single_feature_score=round(score, 4), metric=metric,
        ))

    # -- leakage through a formula ----------------------------------------------
    # Beyond the single suspects already flagged: columns that rebuild the target together.
    suspects = {f["column"] for f in findings if f["check"] == "leakage_suspect"}
    number = formula_target(sample, target, problem_type, suspects)
    if number is not None:
        formula = formula_leak(sample, number, [c for c in features if c not in suspects])
        if formula:
            findings.append(_finding(
                "leakage_formula", "critical",
                f"These columns rebuild the target by plain arithmetic: {describe_formula(target, formula)}. A model "
                "given all of them adds them up instead of predicting; check when each one is known.",
                None, formula=formula,
            ))

    return _report(run_id, target, problem_type, n, df, class_balance, imbalance_info, findings)


def _report(run_id, target, problem_type, n, df, class_balance, imbalance_info, findings) -> dict:
    findings.sort(key=lambda f: SEVERITY_ORDER[f["severity"]])
    report = {
        "target": target,
        "problem_type": problem_type,
        "n_rows": n,
        "n_cols": int(df.shape[1]),
        "class_balance": class_balance,
        "imbalance": imbalance_info,
        "findings": findings,
        "counts": {s: sum(1 for f in findings if f["severity"] == s) for s in SEVERITY_ORDER},
    }
    storage.write_json(run_id, "audit.json", report)
    return report
