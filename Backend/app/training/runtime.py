"""Load and run any model Namtheg trains.

Used by the inference endpoint (app/deploy/inference_app.py) and copied
verbatim as runtime.py into every downloadable model package, so it must stay
self-contained: numpy and pandas at the top, everything else imported only by
the model that needs it.

Models are stored in portable forms, never as library pickles that break
across builds: XGBoost JSON, CatBoost .cbm, plain numpy arrays, and
scikit-learn objects (portable at the pinned version).
"""
import warnings

import numpy as np
import pandas as pd

BUNDLE_FORMAT = "namtheg-bundle-v2"
LEGACY_FORMAT = "namtheg-portable-v1"
DATE_PARTS = ("year", "month", "day", "day_of_week", "hour")


# -- input cleaning --------------------------------------------------------------
# The analyst agent's validated cleaning. Feature engineering applies it to the
# training data and the model applies it to every input, through `prepare`, so
# predictions always see inputs cleaned the way the training data was.

def _number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def is_code(values: pd.Series, codes: list) -> pd.Series:
    """Where a value is one of `codes`: compared as numbers when both sides are
    numeric (so -999 matches -999.0), otherwise as trimmed text."""
    numbers = [n for n in (_number(c) for c in codes) if n is not None]
    hit = values.notna() & values.astype(str).str.strip().isin({str(c).strip() for c in codes})
    if numbers:
        hit |= pd.to_numeric(values, errors="coerce").isin(numbers)
    return hit


def as_code(value):
    """A value as a category label. Whole numbers lose their decimal point, so
    7, 7.0 and "7" are the same category."""
    if pd.isna(value):
        return np.nan
    number = _number(value)
    if number is None:
        return str(value).strip()
    return str(int(number)) if number.is_integer() else str(number)


def parse_dates(values: pd.Series) -> pd.Series:
    """Timestamps from a date column. One format inferred for the whole column
    is fast; values in several formats are parsed one by one instead."""
    if pd.api.types.is_datetime64_any_dtype(values):
        return values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # "could not infer format": the fallback below covers it
        stamps = pd.to_datetime(values, errors="coerce", utc=True)
    if stamps.notna().sum() < 0.9 * values.notna().sum():
        stamps = pd.to_datetime(values, errors="coerce", format="mixed", utc=True)
    return stamps


def date_part(stamps: pd.Series, part: str) -> pd.Series:
    """One calendar part of parsed timestamps, as floats (missing where unparsed)."""
    values = stamps.dt.dayofweek if part == "day_of_week" else getattr(stamps.dt, part)
    return values.astype("float64")


def prepare(X: pd.DataFrame, cleaning: dict | None) -> pd.DataFrame:
    """Apply cleaning to raw input columns: codes that mean missing become
    missing, number codes become categories, and each date column is replaced
    by its calendar parts (new columns named "<column>__<part>")."""
    if not cleaning:
        return X
    X = X.copy()
    for col, codes in (cleaning.get("missing_codes") or {}).items():
        if col in X:
            X[col] = X[col].mask(is_code(X[col], codes))
    for col in cleaning.get("categorical") or []:
        if col in X:
            X[col] = X[col].map(as_code).astype(object)
    for col, parts in (cleaning.get("date_parts") or {}).items():
        if col in X:
            stamps = parse_dates(X[col])
            for part in parts:
                X[f"{col}__{part}"] = date_part(stamps, part)
            X = X.drop(columns=[col])
    return X


# -- estimators ------------------------------------------------------------------

def load_estimator(spec: dict, task: str):
    kind = spec["kind"]
    classification = task == "classification"
    if kind == "xgboost-json":
        import xgboost

        model = xgboost.XGBClassifier() if classification else xgboost.XGBRegressor()
        model.load_model(bytearray(spec["bytes"]))
        return model
    if kind == "catboost-cbm":
        import catboost

        model = catboost.CatBoostClassifier() if classification else catboost.CatBoostRegressor()
        model.load_model(blob=spec["bytes"])
        return model
    if kind == "sklearn":
        return spec["object"]
    raise ValueError(f"Unknown estimator kind {kind!r}")


class TabularModel:
    """Classification/regression: raw columns in, predictions out."""

    def __init__(self, bundle: dict):
        # feature_cols are the raw inputs a prediction takes; model_cols are what
        # they become after cleaning. Older bundles have no cleaning, so the two match.
        self.feature_cols = bundle["feature_cols"]
        self.model_cols = bundle.get("model_cols") or self.feature_cols
        self.cleaning = bundle.get("cleaning")
        self.preprocessor = bundle["preprocessor"]
        self.estimator = load_estimator(bundle["estimator"], bundle["task"])
        if hasattr(self.estimator, "predict_proba"):
            self.predict_proba = self._predict_proba

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.preprocessor.transform(prepare(X[self.feature_cols], self.cleaning)[self.model_cols])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # ravel: CatBoost multiclass predict returns shape (n, 1).
        return np.asarray(self.estimator.predict(self.transform(X))).ravel()

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.estimator.predict_proba(self.transform(X)))


# -- entry point -------------------------------------------------------------------------

_MODELS = {
    "classification": TabularModel,
    "regression": TabularModel,
}


def load_bundle(bundle: dict) -> dict:
    """Turn a loaded model.joblib into {"model": <predictor>, ...}. Bundles from
    before the portable formats already hold a pickled pipeline under "model"."""
    if bundle.get("format") == LEGACY_FORMAT:
        bundle = {
            **bundle,
            "format": BUNDLE_FORMAT,
            "task": bundle["problem_type"],
            "estimator": {"kind": bundle["model_format"], "bytes": bundle["model_bytes"]},
        }
    if bundle.get("format") != BUNDLE_FORMAT:
        return bundle
    # Clustering, forecasting, image and free-text models were removed from Namtheg.
    if bundle["task"] not in _MODELS:
        raise ValueError(f"{bundle['task'].replace('_', ' ').capitalize()} models are no longer supported; "
                         "retrain as classification or regression.")
    if bundle.get("text"):
        raise ValueError("This model embeds free text, which is no longer supported; retrain it.")
    return {**bundle, "model": _MODELS[bundle["task"]](bundle)}
