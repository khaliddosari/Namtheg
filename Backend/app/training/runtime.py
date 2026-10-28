"""Load and run any model Namtheg trains.

Used by the inference endpoint (app/deploy/inference_app.py) and copied
verbatim as runtime.py into every downloadable model package, so it must stay
self-contained: numpy and pandas at the top, everything else imported only by
the model that needs it.

Models are stored in portable forms, never as library pickles that break
across builds: XGBoost JSON, CatBoost .cbm, plain numpy arrays, and
scikit-learn objects (portable at the pinned version).
"""
import ast
import operator
import re
import warnings

import numpy as np
import pandas as pd

BUNDLE_FORMAT = "namtheg-bundle-v2"
LEGACY_FORMAT = "namtheg-portable-v1"
DATE_PARTS = ("year", "month", "day", "day_of_week", "hour")
MAX_FORMULA_LENGTH = 300
MAX_FORMULA_NODES = 60


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


# -- new columns -------------------------------------------------------------------
# A column the user asked for, built from the others. A formula is parsed, never
# executed: only arithmetic, comparisons, and/or/not and a few functions of
# columns and constants are allowed, so a formula can't run code wherever the
# model runs. A column name that isn't a plain identifier goes in backticks.

_ARITHMETIC = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}
_COMPARISONS = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
                ast.Eq: operator.eq, ast.NotEq: operator.ne}
_FUNCTIONS = {"log": np.log, "log1p": np.log1p, "sqrt": np.sqrt, "abs": np.abs}


def _parse_formula(formula: str) -> tuple[ast.AST, dict[str, str]]:
    """The formula's syntax tree, with each `backticked` name swapped for a
    placeholder, and the map from placeholders back to the names."""
    if not isinstance(formula, str) or not formula.strip() or len(formula) > MAX_FORMULA_LENGTH:
        raise ValueError(f"A formula is an expression of at most {MAX_FORMULA_LENGTH} characters.")
    names: dict[str, str] = {}

    def placeholder(match: re.Match) -> str:
        key = f"_col{len(names)}_"
        names[key] = match.group(1)
        return key

    try:
        tree = ast.parse(re.sub(r"`([^`]+)`", placeholder, formula.strip()), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"The formula doesn't parse: {e.msg}.") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_FORMULA_NODES:
        raise ValueError("The formula is too long.")
    return tree, names


def _columns_in(node: ast.AST, names: dict[str, str], found: list[str]) -> None:
    """Collects the columns a formula reads, and refuses anything else."""
    if isinstance(node, ast.Expression):
        _columns_in(node.body, names, found)
    elif isinstance(node, ast.BinOp) and type(node.op) in _ARITHMETIC:
        _columns_in(node.left, names, found)
        _columns_in(node.right, names, found)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
        _columns_in(node.operand, names, found)
    elif isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in _COMPARISONS:
        _columns_in(node.left, names, found)
        _columns_in(node.comparators[0], names, found)
    elif isinstance(node, ast.BoolOp):
        for value in node.values:
            _columns_in(value, names, found)
    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS
          and len(node.args) == 1 and not node.keywords):
        _columns_in(node.args[0], names, found)
    elif isinstance(node, ast.Name):
        found.append(names.get(node.id, node.id))
    elif not (isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str))
              and not isinstance(node.value, bool)):
        raise ValueError(f"A formula may only use columns, numbers, text, + - * / **, comparisons, and, or, not "
                         f"and {', '.join(_FUNCTIONS)}; it has {type(node).__name__}.")


def new_column_sources(spec: dict) -> list[str]:
    """The columns a new column is built from."""
    if "bins" in spec:
        return [spec["bins"]["column"]]
    found: list[str] = []
    _columns_in(*_parse_formula(spec["formula"]), found)
    return list(dict.fromkeys(found))


def _numeric(value):
    if isinstance(value, pd.Series):
        return pd.to_numeric(value, errors="coerce").astype("float64")
    number = _number(value)
    return np.nan if number is None else number


def _evaluate(node: ast.AST, X: pd.DataFrame, names: dict[str, str]):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, X, names)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return X[names.get(node.id, node.id)]
    if isinstance(node, ast.BinOp):
        return _ARITHMETIC[type(node.op)](_numeric(_evaluate(node.left, X, names)),
                                          _numeric(_evaluate(node.right, X, names)))
    if isinstance(node, ast.UnaryOp):
        value = _numeric(_evaluate(node.operand, X, names))
        return 1.0 - value if isinstance(node.op, ast.Not) else -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Call):
        return _FUNCTIONS[node.func.id](_numeric(_evaluate(node.args[0], X, names)))
    if isinstance(node, ast.BoolOp):
        values = [_numeric(_evaluate(v, X, names)) for v in node.values]
        combine = np.minimum if isinstance(node.op, ast.And) else np.maximum
        result = values[0]
        for value in values[1:]:
            result = combine(result, value)
        return result
    # A comparison is 1 or 0, and missing where a side is. Against text it
    # compares trimmed text, so a category column can be tested for a value.
    left, right = _evaluate(node.left, X, names), _evaluate(node.comparators[0], X, names)
    compare = _COMPARISONS[type(node.ops[0])]
    if isinstance(left, str) or isinstance(right, str):
        def as_text(v):
            return v.astype(str).str.strip().where(v.notna()) if isinstance(v, pd.Series) else str(v).strip()
        left, right = as_text(left), as_text(right)
    else:
        left, right = _numeric(left), _numeric(right)
    missing = pd.Series(False, index=X.index)
    for side in (left, right):
        if isinstance(side, pd.Series):
            missing |= side.isna()
    return pd.Series(compare(left, right), index=X.index).astype("float64").mask(missing)


def build_new_column(X: pd.DataFrame, spec: dict) -> pd.Series:
    """The values of one new column: a number from a formula (missing where it
    can't be computed, such as a division by zero), or the label of the range a
    numeric column falls in, ranges starting at each cut point."""
    if "bins" in spec:
        bins = spec["bins"]
        values = pd.to_numeric(X[bins["column"]], errors="coerce")
        ranges = pd.cut(values, [-np.inf, *bins["edges"], np.inf], labels=bins["labels"], right=False)
        return pd.Series(ranges, index=X.index).astype(object).where(values.notna())
    tree, names = _parse_formula(spec["formula"])
    _columns_in(tree, names, [])  # refuses anything but the allowed forms, wherever the spec came from
    with np.errstate(all="ignore"):
        result = pd.Series(_evaluate(tree, X, names), index=X.index)
    return pd.to_numeric(result, errors="coerce").astype("float64").replace([np.inf, -np.inf], np.nan)


def prepare(X: pd.DataFrame, cleaning: dict | None) -> pd.DataFrame:
    """Apply cleaning to raw input columns: codes that mean missing become
    missing, the new columns the user asked for are built from them, number
    codes become categories, and each date column is replaced by its calendar
    parts (new columns named "<column>__<part>")."""
    if not cleaning:
        return X
    X = X.copy()
    for col, codes in (cleaning.get("missing_codes") or {}).items():
        if col in X:
            X[col] = X[col].mask(is_code(X[col], codes))
    for spec in cleaning.get("new_columns") or []:
        if all(c in X for c in new_column_sources(spec)):
            X[spec["name"]] = build_new_column(X, spec)
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
