"""Model training engine. Production runs it only inside the GPU container
(app/training/gpu_app.py), with device="cuda".

Self-contained on purpose: it imports third-party packages only, never the
rest of `app`, so the GPU image needs nothing else from the backend.

What one call does, all on the same held-out split and CV folds:
1. Encode labels, split 80/20 (stratified for classification).
2. Score a feature-blind baseline.
3. Cross-validate every GPU candidate; the imbalance plan (class weights and
   the selection metric) is applied inside each training fold.
4. Tune the best candidate with Optuna, selecting by CV mean only.
5. Fit the winner on the training split, evaluate once on the test split.
6. Return metrics, predictions, the portable model bundle, and native weights.

Everything returned is plain Python (lists, floats, str, bytes), so the
backend can unpickle the result whatever its own numpy version.
"""
import io
import json
import logging
import os
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils.class_weight import compute_sample_weight

log = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FOLDS = 5
MIN_ROWS_PER_CLASS = 3
ONE_HOT_MAX_CARDINALITY = 10

# Three GPU-native tree learners that grow trees differently: depth-wise
# (XGBoost), leaf-wise (XGBoost with lossguide, LightGBM-style), and
# symmetric/oblivious (CatBoost).
CANDIDATES = ("XGBoost", "XGBoost Leaf-wise", "CatBoost")

DEFAULT_PARAMS = {
    "XGBoost": {"n_estimators": 400, "learning_rate": 0.08, "max_depth": 6, "subsample": 0.9,
                "colsample_bytree": 0.9, "min_child_weight": 1.0, "reg_lambda": 1.0},
    "XGBoost Leaf-wise": {"n_estimators": 400, "learning_rate": 0.08, "max_leaves": 31, "subsample": 0.9,
                          "colsample_bytree": 0.9, "min_child_weight": 1.0, "reg_lambda": 1.0},
    "CatBoost": {"iterations": 600, "learning_rate": 0.08, "depth": 6, "l2_leaf_reg": 3.0},
}
# Settings that define a candidate rather than tune it.
FIXED_PARAMS = {"XGBoost Leaf-wise": {"grow_policy": "lossguide", "max_depth": 0}}


def _search_space(name: str, trial) -> dict:
    if name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "XGBoost Leaf-wise":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1200, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_leaves": trial.suggest_int("max_leaves", 8, 256, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    return {
        "iterations": trial.suggest_int("iterations", 200, 1500, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    }


# -- preprocessing -------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Numbers pass through untouched (the tree models handle missing values
    natively, and missingness can carry signal). Low-cardinality text is
    one-hot encoded, high-cardinality text ordinal encoded; unseen categories
    at prediction time become all-zeros / -1 instead of an error."""
    numeric, one_hot, ordinal = [], [], []
    for c in X.columns:
        if is_numeric_dtype(X[c]):
            numeric.append(c)
        elif X[c].nunique(dropna=True) <= ONE_HOT_MAX_CARDINALITY:
            one_hot.append(c)
        else:
            ordinal.append(c)
    transformers: list = []
    if numeric:
        transformers.append(("num", "passthrough", numeric))
    if one_hot:
        transformers.append(("cat_low", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary")),
        ]), one_hot))
    if ordinal:
        transformers.append(("cat_high", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), ordinal))
    return ColumnTransformer(transformers, remainder="drop")


def describe_preprocessor(pre: ColumnTransformer) -> dict:
    """What a fitted preprocessor does, for anyone using the native weights."""
    out: dict = {"numeric_passthrough": [], "one_hot": {}, "ordinal": {},
                 "encoded_feature_names": [str(n) for n in pre.get_feature_names_out()]}
    for name, trans, cols in pre.transformers_:
        if name == "num":
            out["numeric_passthrough"] = list(cols)
        elif name in ("cat_low", "cat_high"):
            encoder = trans.named_steps["encoder"]
            key = "one_hot" if name == "cat_low" else "ordinal"
            for col, cats in zip(cols, encoder.categories_):
                out[key][col] = [c.item() if hasattr(c, "item") else c for c in cats]
    return out


# -- models ----------------------------------------------------------------------

def make_model(name: str, problem_type: str, params: dict, device: str, balanced: bool):
    classification = problem_type == "classification"
    params = {**params, **FIXED_PARAMS.get(name, {})}
    if name in ("XGBoost", "XGBoost Leaf-wise"):
        import xgboost as xgb

        common = {"device": device, "tree_method": "hist", "random_state": RANDOM_STATE, **params}
        return xgb.XGBClassifier(**common) if classification else xgb.XGBRegressor(**common)
    if name == "CatBoost":
        import catboost

        common = {"task_type": "GPU" if device.startswith("cuda") else "CPU", "random_seed": RANDOM_STATE,
                  "verbose": 0, "allow_writing_files": False, **params}
        if classification:
            if balanced:
                common["auto_class_weights"] = "Balanced"
            return catboost.CatBoostClassifier(**common)
        return catboost.CatBoostRegressor(**common)
    raise ValueError(f"Unknown model {name!r}")


def fit_pipeline(name, problem_type, params, device, balanced, X, y) -> Pipeline:
    pipe = Pipeline([("preprocessor", build_preprocessor(X)),
                     ("model", make_model(name, problem_type, params, device, balanced))])
    fit_params = {}
    # XGBoost takes per-row weights; CatBoost got auto_class_weights above.
    # Computed from this fold's own labels only.
    if balanced and name != "CatBoost":
        fit_params["model__sample_weight"] = compute_sample_weight("balanced", y)
    pipe.fit(X, y, **fit_params)
    return pipe


def predict(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(pipe.predict(X)).ravel()  # CatBoost multiclass returns (n, 1)


# -- scoring -------------------------------------------------------------------------

def score(metric: str, y_true, y_pred) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return float(r2_score(y_true, y_pred))


def full_metrics(problem_type: str, y_true, y_pred, proba=None) -> dict:
    if problem_type != "classification":
        return {
            "r2": float(r2_score(y_true, y_pred)),
            "rmse": float(root_mean_squared_error(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
        }
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if proba is not None and proba.shape[1] == 2 and len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
        out["pr_auc"] = float(average_precision_score(y_true, proba[:, 1]))
    return out


def make_folds(problem_type: str, y_train: np.ndarray) -> list:
    if problem_type == "classification":
        n = max(2, min(MAX_FOLDS, int(pd.Series(y_train).value_counts().min())))
        splitter = StratifiedKFold(n_splits=n, shuffle=True, random_state=RANDOM_STATE)
    else:
        n = max(2, min(MAX_FOLDS, len(y_train) // 10))
        splitter = KFold(n_splits=n, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(np.zeros(len(y_train)), y_train))


def cross_validate(name, params, X, y, folds, problem_type, device, balanced, metric) -> list[float]:
    scores = []
    for train_idx, val_idx in folds:
        pipe = fit_pipeline(name, problem_type, params, device, balanced, X.iloc[train_idx], y[train_idx])
        scores.append(score(metric, y[val_idx], predict(pipe, X.iloc[val_idx])))
    return scores


def baseline(problem_type, X_train, y_train, y_test, folds, metric) -> dict:
    """Predict the training majority class (or mean) for every row; no features used."""
    def constant(y_fit):
        if problem_type == "classification":
            return pd.Series(y_fit).value_counts().index[0]
        return float(np.mean(y_fit))

    cv = [score(metric, y_train[v], np.full(len(v), constant(y_train[t]))) for t, v in folds]
    test = score(metric, y_test, np.full(len(y_test), constant(y_train)))
    return {
        "name": "Majority class" if problem_type == "classification" else "Mean of target",
        "metric": metric,
        "cv_mean": round(float(np.mean(cv)), 4),
        "test_score": round(test, 4),
    }


def feature_importances(pipe: Pipeline) -> list[dict]:
    est = pipe[-1]
    imps = np.asarray(getattr(est, "feature_importances_", []), dtype=float)
    if imps.size == 0:
        return []
    names = [str(n) for n in pipe[:-1].get_feature_names_out()]
    if len(names) != len(imps):
        names = [f"feature_{i}" for i in range(len(imps))]
    if imps.sum() > 0:
        imps = imps / imps.sum()  # CatBoost reports percentages, XGBoost fractions
    pairs = sorted(zip(names, imps.tolist()), key=lambda kv: kv[1], reverse=True)[:10]
    return [{"feature": f, "importance": round(float(i), 4)} for f, i in pairs]


def native_weights(name: str, pipe: Pipeline) -> tuple[str, bytes]:
    """The trained trees in the library's own portable format (XGBoost JSON
    keeps the scikit-learn wrapper's attributes, e.g. the number of classes)."""
    est = pipe[-1]
    with tempfile.TemporaryDirectory() as d:
        if name == "CatBoost":
            path, fmt = os.path.join(d, "model.cbm"), "catboost-cbm"
            est.save_model(path, format="cbm")
        else:
            path, fmt = os.path.join(d, "model.json"), "xgboost-json"
            est.save_model(path)
        with open(path, "rb") as f:
            return fmt, f.read()


BUNDLE_FORMAT = "namtheg-portable-v1"


def portable_bundle(name, preprocessor, model_format, model_bytes, feature_cols, problem_type, class_labels) -> bytes:
    """What model.joblib holds. The model is stored in its native format, not
    pickled: XGBoost's pickle is a memory snapshot that fails to load on a
    different build or OS even at the same version, while its JSON format is
    portable. Only the scikit-learn preprocessor is pickled. load_bundle()
    rebuilds the pipeline."""
    buf = io.BytesIO()
    joblib.dump({
        "format": BUNDLE_FORMAT,
        "preprocessor": preprocessor,
        "model_format": model_format,
        "model_bytes": model_bytes,
        "feature_cols": feature_cols,
        "problem_type": problem_type,
        "model_name": name,
        "class_labels": class_labels,
    }, buf)
    return buf.getvalue()


def load_bundle(bundle: dict) -> dict:
    """Turn a loaded model.joblib into {"model": <pipeline>, ...}.

    Shared by the inference endpoint and copied verbatim into the downloadable
    package's predict.py, so it must stay self-contained (imports inside).
    Bundles from before the portable format hold the pickled pipeline already.
    """
    if bundle.get("format") != "namtheg-portable-v1":
        return bundle
    from sklearn.pipeline import Pipeline

    classification = bundle["problem_type"] == "classification"
    if bundle["model_format"] == "xgboost-json":
        import xgboost

        model = xgboost.XGBClassifier() if classification else xgboost.XGBRegressor()
        model.load_model(bytearray(bundle["model_bytes"]))
    elif bundle["model_format"] == "catboost-cbm":
        import catboost

        model = catboost.CatBoostClassifier() if classification else catboost.CatBoostRegressor()
        model.load_model(blob=bundle["model_bytes"])
    else:
        raise ValueError(f"Unknown model format {bundle['model_format']!r}")
    return {**bundle, "model": Pipeline([("preprocessor", bundle["preprocessor"]), ("model", model)])}


# -- tuning --------------------------------------------------------------------------

def tune(name, X, y, folds, problem_type, device, balanced, metric, start_score, n_trials, timeout) -> tuple[dict, float, list]:
    """Optuna TPE search around the champion. Returns (best params, best CV
    mean, trial log). The defaults win unless a trial beats them on CV mean."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    best = {"params": dict(DEFAULT_PARAMS[name]), "score": start_score}
    log_rows = [{"trial": 0, "parameters": "Baseline Settings", "score": round(start_score, 4),
                 "result": "Champion defaults (CV mean)"}]

    def objective(trial):
        params = _search_space(name, trial)
        try:
            value = float(np.mean(cross_validate(name, params, X, y, folds, problem_type, device, balanced, metric)))
        except Exception as e:
            log_rows.append({"trial": trial.number + 1, "parameters": json.dumps(params), "score": None,
                             "result": f"Failed: {str(e)[:80]}"})
            raise
        delta = value - best["score"]
        if value > best["score"]:
            best.update(params=params, score=value)
            result = f"New champion! CV +{delta:.4f}"
        else:
            result = f"No improvement (CV={value:.4f})"
        log_rows.append({"trial": trial.number + 1, "parameters": json.dumps(params), "score": round(value, 4),
                         "result": result})
        return value

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, catch=(Exception,))
    return best["params"], best["score"], log_rows


# -- entry point -------------------------------------------------------------------

def run_training(data: bytes, target: str, problem_type: str, plan: dict, device: str) -> dict:
    """plan: {"imbalance": <app.pipeline.imbalance.assess output>,
              "tuning_trials": int, "tuning_timeout_seconds": int}"""
    started = time.monotonic()
    df = pd.read_parquet(io.BytesIO(data))
    X = df.drop(columns=[target])
    feature_cols = X.columns.tolist()

    imbalance = plan.get("imbalance") or {}
    classification = problem_type == "classification"
    balanced = bool(classification and imbalance.get("strategy") == "balanced_class_weights")
    metric = imbalance.get("selection_metric") or ("accuracy" if classification else "r2")

    class_labels = None
    if classification:
        classes, y = np.unique(df[target].to_numpy(), return_inverse=True)
        class_labels = [c.item() if hasattr(c, "item") else c for c in classes]
        too_small = [str(class_labels[i]) for i, n in enumerate(np.bincount(y)) if n < MIN_ROWS_PER_CLASS]
        if too_small:
            raise ValueError(
                f"Classes {too_small} have fewer than {MIN_ROWS_PER_CLASS} rows, too few to appear in the test set "
                "and in every cross-validation training fold. Collect more rows or merge rare classes."
            )
    else:
        y = df[target].to_numpy(dtype=float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y if classification else None,
    )
    if classification and np.bincount(y_train, minlength=len(class_labels)).min() < 2:
        raise ValueError("After the train/test split a class has fewer than 2 training rows; collect more data.")
    folds = make_folds(problem_type, y_train)
    base = baseline(problem_type, X_train, y_train, y_test, folds, metric)

    all_models = []
    for name in CANDIDATES:
        cv = cross_validate(name, DEFAULT_PARAMS[name], X_train, y_train, folds, problem_type, device, balanced, metric)
        all_models.append({"name": name, "cv_mean": round(float(np.mean(cv)), 4), "cv_std": round(float(np.std(cv)), 4),
                           "cv_folds": [round(s, 4) for s in cv]})
    all_models.sort(key=lambda m: m["cv_mean"], reverse=True)
    champion = all_models[0]["name"]

    tuned_params, cv_mean, trials = tune(
        champion, X_train, y_train, folds, problem_type, device, balanced, metric,
        start_score=all_models[0]["cv_mean"],
        n_trials=int(plan.get("tuning_trials", 20)),
        timeout=int(plan.get("tuning_timeout_seconds", 300)),
    )

    final = fit_pipeline(champion, problem_type, tuned_params, device, balanced, X_train, y_train)
    test_pred = predict(final, X_test)
    train_pred = predict(final, X_train)
    proba = final.predict_proba(X_test) if classification else None
    test_metrics = full_metrics(problem_type, y_test, test_pred, proba)
    train_metrics = full_metrics(problem_type, y_train, train_pred)

    weights_format, weights = native_weights(champion, final)
    bundle_bytes = portable_bundle(champion, final[0], weights_format, weights, feature_cols, problem_type, class_labels)

    return {
        "problem_type": problem_type,
        "model_name": champion,
        "selection_metric": metric,
        "cv_mean": round(float(cv_mean), 4),
        "params": tuned_params,
        "all_models": all_models,
        "baseline": base,
        "tuning_trials": trials,
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "train_score": score(metric, y_train, train_pred),
        "test_score": score(metric, y_test, test_pred),
        "n_folds": len(folds),
        "test_size": int(len(y_test)),
        "n_classes": len(class_labels) if class_labels else None,
        "class_labels": class_labels,
        "imbalance_applied": {"class_weights": balanced, "selection_metric": metric},
        "feature_cols": feature_cols,
        "top_features": feature_importances(final),
        "preprocessing": describe_preprocessor(final[0]),
        "y_test": y_test.tolist(),
        "y_pred": test_pred.tolist(),
        "bundle_bytes": bundle_bytes,
        "weights_format": weights_format,
        "weights_bytes": weights,
        "device": device,
        "library_versions": library_versions(),
        "seconds": round(time.monotonic() - started, 1),
    }


def library_versions() -> dict:
    """Exact versions the model was trained with; loading the pickled pipeline
    elsewhere needs the same ones."""
    import platform

    import catboost
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "xgboost": xgboost.__version__,
        "catboost": catboost.__version__,
    }
