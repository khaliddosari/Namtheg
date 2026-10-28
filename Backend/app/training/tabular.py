"""Supervised tabular training (classification and regression) on the GPU.

One call trains one *group* of candidates; the backend runs groups in
parallel on separate GPUs. Within a group, on the same held-out split and CV
folds:
1. Every candidate is cross-validated with default parameters.
2. The group's best is tuned with Optuna, selected by CV mean only. It first
   tries STARTING_POINTS, settings that did well across many datasets, then
   TPE steers from the results. Weak trials stop early: on large folds after
   one fold unless among the best third so far (successive halving), on small
   folds once behind the median. The budget counts full CV checks, so a trial
   stopped early costs only the folds it used, and the search ends once
   TUNING_PATIENCE checks' worth of folds fail to beat the best by more than
   the noise in its CV folds.
3. It is refit on the training split and scored once on the test split.

Once the group's time budget is spent it starts no new candidate or tuning
trial and goes straight to step 3 with what it has. Progress goes to the GPU
service's logs.

Gradient-boosted trees use early stopping on a slice of each training fold
instead of a guessed tree count. SVM, KNN and linear models are ordinary
scikit-learn estimators that cuml.accel runs on the GPU; `gpu_only` rejects
any that fall back to CPU.
"""
import io
import json
import os
import tempfile
import time

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from app.training.common import (
    MIN_ROWS_PER_CLASS,
    RANDOM_STATE,
    TEST_SIZE,
    full_metrics,
    gpu_only,
    library_versions,
    make_folds,
    python_value,
    score,
)
from app.training.runtime import BUNDLE_FORMAT

ONE_HOT_MAX_CARDINALITY = 10
SCALED_MAX_CATEGORIES = 20
MAX_TREES = 3000
MIN_LEARNING_RATE = 0.015  # slower boosted trees need more than MAX_TREES trees on a large table
EARLY_STOPPING_ROUNDS = 50
EARLY_STOPPING_FRACTION = 0.1
SVM_MAX_ROWS = 100_000  # kernel SVM cost grows quadratically; past this it isn't worth the GPU time
KNN_MAX_ROWS = 1_000_000
# Full CV checks' worth of folds, after the starting points, without a gain
# beyond CV noise before tuning stops.
TUNING_PATIENCE = 5
# A fold this large ranks settings reliably on its own, so a weak trial can be
# dropped after one; smaller folds wait for the median.
HALVING_MIN_FOLD_ROWS = 10_000

GBDT = ("XGBoost", "XGBoost Leaf-wise", "CatBoost")
FAMILY = {"XGBoost": "gbdt", "XGBoost Leaf-wise": "gbdt", "CatBoost": "gbdt",
          "SVM": "kernel", "KNN": "neighbors", "Linear": "linear"}
# Groups run in parallel, one GPU container each.
GROUPS = (GBDT, ("SVM", "KNN", "Linear"))

DEFAULT_PARAMS = {
    "XGBoost": {"learning_rate": 0.05, "max_depth": 6, "subsample": 0.9, "colsample_bytree": 0.9,
                "min_child_weight": 1.0, "reg_lambda": 1.0},
    "XGBoost Leaf-wise": {"learning_rate": 0.05, "max_leaves": 31, "subsample": 0.9, "colsample_bytree": 0.9,
                          "min_child_weight": 1.0, "reg_lambda": 1.0},
    "CatBoost": {"learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 3.0},
    "SVM": {"C": 1.0, "gamma": "scale"},
    "KNN": {"n_neighbors": 15, "weights": "distance"},
    "Linear": {"strength": 1.0},
}
# Settings that define a candidate rather than tune it.
FIXED_PARAMS = {"XGBoost Leaf-wise": {"grow_policy": "lossguide", "max_depth": 0}}
# Tried first when a model is tuned, before the search steers itself. The
# boosted-tree ones are the top entries of AutoGluon's 2023 zero-shot portfolio
# (Apache-2.0), picked for doing well across a large benchmark of datasets, in
# its priority order; settings an entry leaves out are the library defaults
# AutoGluon ran it with. Leaf-wise XGBoost takes the LightGBM entries, which
# also grow trees leaf by leaf (num_leaves -> max_leaves, feature_fraction ->
# colsample_bytree), except extra-trees ones, which XGBoost can't do. Entries
# slower than MIN_LEARNING_RATE are left out. SVM, KNN and linear models have
# no such portfolio, so theirs are spread across the search range.
STARTING_POINTS = {
    "XGBoost": [
        {"learning_rate": 0.0181, "max_depth": 10, "subsample": 1.0, "colsample_bytree": 0.692,
         "min_child_weight": 0.603, "reg_lambda": 1.0},
        {"learning_rate": 0.0878, "max_depth": 5, "subsample": 1.0, "colsample_bytree": 0.663,
         "min_child_weight": 0.629, "reg_lambda": 1.0},
        {"learning_rate": 0.0929, "max_depth": 7, "subsample": 1.0, "colsample_bytree": 0.909,
         "min_child_weight": 0.804, "reg_lambda": 1.0},
        {"learning_rate": 0.0384, "max_depth": 7, "subsample": 1.0, "colsample_bytree": 0.745,
         "min_child_weight": 0.556, "reg_lambda": 1.0},
        {"learning_rate": 0.0779, "max_depth": 6, "subsample": 1.0, "colsample_bytree": 0.633,
         "min_child_weight": 1.08, "reg_lambda": 1.0},
    ],
    "XGBoost Leaf-wise": [
        {"learning_rate": 0.03, "max_leaves": 128, "subsample": 1.0, "colsample_bytree": 0.9,
         "min_child_weight": 1.0, "reg_lambda": 1.0},
        {"learning_rate": 0.0471, "max_leaves": 89, "subsample": 1.0, "colsample_bytree": 0.625,
         "min_child_weight": 1.0, "reg_lambda": 1.0},
        {"learning_rate": 0.0186, "max_leaves": 22, "subsample": 1.0, "colsample_bytree": 0.742,
         "min_child_weight": 1.0, "reg_lambda": 1.0},
        {"learning_rate": 0.0313, "max_leaves": 210, "subsample": 1.0, "colsample_bytree": 0.825,
         "min_child_weight": 1.0, "reg_lambda": 1.0},
    ],
    "CatBoost": [
        {"learning_rate": 0.0686, "depth": 6, "l2_leaf_reg": 2.15},
        {"learning_rate": 0.0494, "depth": 4, "l2_leaf_reg": 4.56},
        {"learning_rate": 0.0173, "depth": 8, "l2_leaf_reg": 3.33},
        {"learning_rate": 0.0503, "depth": 5, "l2_leaf_reg": 1.05},
        {"learning_rate": 0.0848, "depth": 7, "l2_leaf_reg": 4.52},
    ],
    "SVM": [{"C": 10.0, "gamma": 0.01}, {"C": 1.0, "gamma": 0.1}, {"C": 100.0, "gamma": 0.001}],
    "KNN": [{"n_neighbors": 5, "weights": "distance"}, {"n_neighbors": 30, "weights": "distance"},
            {"n_neighbors": 15, "weights": "uniform"}],
    "Linear": [{"strength": 0.01}, {"strength": 0.1}, {"strength": 10.0}],
}


def search_space(name: str, trial) -> dict:
    f = trial.suggest_float
    if name == "XGBoost":
        return {"learning_rate": f("learning_rate", MIN_LEARNING_RATE, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": f("subsample", 0.6, 1.0), "colsample_bytree": f("colsample_bytree", 0.5, 1.0),
                "min_child_weight": f("min_child_weight", 0.5, 10.0, log=True),
                "reg_lambda": f("reg_lambda", 1e-3, 10.0, log=True)}
    if name == "XGBoost Leaf-wise":
        return {"learning_rate": f("learning_rate", MIN_LEARNING_RATE, 0.3, log=True),
                "max_leaves": trial.suggest_int("max_leaves", 8, 256, log=True),
                "subsample": f("subsample", 0.6, 1.0), "colsample_bytree": f("colsample_bytree", 0.5, 1.0),
                "min_child_weight": f("min_child_weight", 0.5, 10.0, log=True),
                "reg_lambda": f("reg_lambda", 1e-3, 10.0, log=True)}
    if name == "CatBoost":
        return {"learning_rate": f("learning_rate", MIN_LEARNING_RATE, 0.3, log=True),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": f("l2_leaf_reg", 1.0, 10.0, log=True)}
    if name == "SVM":
        return {"C": f("C", 1e-2, 1e2, log=True), "gamma": f("gamma", 1e-4, 1.0, log=True)}
    if name == "KNN":
        return {"n_neighbors": trial.suggest_int("n_neighbors", 3, 60, log=True),
                "weights": trial.suggest_categorical("weights", ["uniform", "distance"])}
    return {"strength": f("strength", 1e-3, 1e2, log=True)}


# -- preprocessing ---------------------------------------------------------------

def build_preprocessor(X: pd.DataFrame, kind: str) -> ColumnTransformer:
    """kind="tree": numbers pass through untouched (boosted trees handle
    missing values natively, and missingness can carry signal); text is
    one-hot or ordinal encoded. kind="scaled" (SVM, KNN, linear): numbers are
    median-imputed and standardised, and text is one-hot encoded with rare
    categories pooled, since distances over ordinal codes mean nothing."""
    numeric = [c for c in X.columns if is_numeric_dtype(X[c])]
    text = [c for c in X.columns if c not in numeric]
    impute = ("imputer", SimpleImputer(strategy="most_frequent"))
    transformers: list = []
    if kind == "tree":
        low = [c for c in text if X[c].nunique(dropna=True) <= ONE_HOT_MAX_CARDINALITY]
        high = [c for c in text if c not in low]
        if numeric:
            transformers.append(("num", "passthrough", numeric))
        if low:
            transformers.append(("cat_low", Pipeline([impute, ("encoder", OneHotEncoder(
                handle_unknown="ignore", sparse_output=False, drop="if_binary"))]), low))
        if high:
            transformers.append(("cat_high", Pipeline([impute, ("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1))]), high))
    else:
        if numeric:
            transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                                  ("scaler", StandardScaler())]), numeric))
        if text:
            transformers.append(("cat_low", Pipeline([impute, ("encoder", OneHotEncoder(
                handle_unknown="infrequent_if_exist", max_categories=SCALED_MAX_CATEGORIES,
                sparse_output=False))]), text))
    return ColumnTransformer(transformers, remainder="drop")


def describe_preprocessor(pre: ColumnTransformer) -> dict:
    """What a fitted preprocessor does, for anyone using the native weights."""
    out: dict = {"numeric": {"columns": [], "standardised": False}, "one_hot": {}, "ordinal": {},
                 "encoded_feature_names": [str(n) for n in pre.get_feature_names_out()]}
    for name, trans, cols in pre.transformers_:
        if name == "num":
            out["numeric"] = {"columns": list(cols), "standardised": isinstance(trans, Pipeline)}
        elif name in ("cat_low", "cat_high"):
            encoder = trans.named_steps["encoder"]
            key = "one_hot" if name == "cat_low" else "ordinal"
            for col, cats in zip(cols, encoder.categories_):
                out[key][col] = [python_value(c) for c in cats]
    return out


# -- estimators ------------------------------------------------------------------

def make_estimator(name, task, params, device, balanced, n_classes, n_estimators=None):
    """n_estimators=None means "use early stopping" for boosted trees."""
    classification = task == "classification"
    params = {**params, **FIXED_PARAMS.get(name, {})}
    if name in ("XGBoost", "XGBoost Leaf-wise"):
        import xgboost as xgb

        common = {"device": device, "tree_method": "hist", "random_state": RANDOM_STATE, **params}
        if n_estimators is None:
            common.update(n_estimators=MAX_TREES, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        else:
            common["n_estimators"] = n_estimators
        return xgb.XGBClassifier(**common) if classification else xgb.XGBRegressor(**common)
    if name == "CatBoost":
        import catboost

        common = {"task_type": "GPU" if device.startswith("cuda") else "CPU", "random_seed": RANDOM_STATE,
                  "verbose": 0, "allow_writing_files": False, "iterations": n_estimators or MAX_TREES, **params}
        if classification:
            if balanced:
                common["auto_class_weights"] = "Balanced"
            return catboost.CatBoostClassifier(**common)
        return catboost.CatBoostRegressor(**common)
    if name == "SVM":
        from sklearn.svm import SVC, SVR

        if not classification:
            # SVR's epsilon-tube is in target units; standardise the target so
            # the defaults mean the same thing for prices in thousands or rates in [0, 1].
            from sklearn.compose import TransformedTargetRegressor

            return TransformedTargetRegressor(regressor=SVR(kernel="rbf", C=params["C"], gamma=params["gamma"]),
                                              transformer=StandardScaler())
        svc = SVC(kernel="rbf", C=params["C"], gamma=params["gamma"],
                  class_weight="balanced" if balanced else None)
        if n_classes > 2:
            # cuml.accel only runs binary SVC on GPU; one-vs-rest keeps every
            # sub-problem binary.
            from sklearn.multiclass import OneVsRestClassifier

            return OneVsRestClassifier(svc)
        return svc
    if name == "KNN":
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

        cls = KNeighborsClassifier if classification else KNeighborsRegressor
        return cls(n_neighbors=params["n_neighbors"], weights=params["weights"], metric="euclidean")
    if name == "Linear":
        from sklearn.linear_model import LogisticRegression, Ridge

        if classification:
            return LogisticRegression(C=params["strength"], max_iter=2000,
                                      class_weight="balanced" if balanced else None)
        return Ridge(alpha=params["strength"])
    raise ValueError(f"Unknown model {name!r}")


def _inner_split(y: np.ndarray, classification: bool):
    stratify = y if classification and pd.Series(y).value_counts().min() >= 2 else None
    return train_test_split(np.arange(len(y)), test_size=EARLY_STOPPING_FRACTION,
                            random_state=RANDOM_STATE, stratify=stratify)


def fit(name, task, params, device, balanced, n_classes, X, y, n_estimators=None):
    """Fit one candidate. Returns (pipeline, trees used). Boosted trees with
    n_estimators=None early-stop on a held-back slice of X."""
    classification = task == "classification"
    kind = "tree" if FAMILY[name] == "gbdt" else "scaled"
    pre = build_preprocessor(X, kind)
    weights = compute_sample_weight("balanced", y) if balanced and name in ("XGBoost", "XGBoost Leaf-wise") else None
    est = make_estimator(name, task, params, device, balanced, n_classes, n_estimators)
    trees = n_estimators
    if FAMILY[name] == "gbdt" and n_estimators is None:
        fit_idx, es_idx = _inner_split(y, classification)
        pre.fit(X.iloc[fit_idx])
        A, B = pre.transform(X.iloc[fit_idx]), pre.transform(X.iloc[es_idx])
        if name == "CatBoost":
            est.fit(A, y[fit_idx], eval_set=(B, y[es_idx]), early_stopping_rounds=EARLY_STOPPING_ROUNDS)
            trees = int(est.get_best_iteration()) + 1
        else:
            est.fit(A, y[fit_idx], sample_weight=None if weights is None else weights[fit_idx],
                    eval_set=[(B, y[es_idx])], verbose=False)
            trees = int(est.best_iteration) + 1
    else:
        A = pre.fit_transform(X)
        if weights is not None:
            est.fit(A, y, sample_weight=weights)
        else:
            est.fit(A, y)
    return Pipeline([("preprocessor", pre), ("model", est)]), trees


def predict(pipe: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(pipe.predict(X)).ravel()  # CatBoost multiclass returns (n, 1)


def cross_validate(name, params, X, y, folds, task, device, balanced, n_classes, metric, trial=None) -> list[float]:
    """CV scores; with an Optuna trial, reports the running mean after each
    fold (step = folds done) and stops early when the trial's pruner says so."""
    import optuna

    scores = []
    for i, (tr, va) in enumerate(folds):
        with gpu_only(device):
            pipe, _ = fit(name, task, params, device, balanced, n_classes, X.iloc[tr], y[tr])
            pred = predict(pipe, X.iloc[va])
        scores.append(score(metric, y[va], pred))
        if trial is not None:
            trial.report(float(np.mean(scores)), i + 1)
            if trial.should_prune():
                raise optuna.TrialPruned(f"after {i + 1} of {len(folds)} folds")
    return scores


def baseline(task, y_train, y_test, folds, metric) -> dict:
    """Predict the training majority class (or mean) for every row; no features used."""
    def constant(y_fit):
        if task == "classification":
            return pd.Series(y_fit).value_counts().index[0]
        return float(np.mean(y_fit))

    cv = [score(metric, y_train[v], np.full(len(v), constant(y_train[t]))) for t, v in folds]
    test = score(metric, y_test, np.full(len(y_test), constant(y_train)))
    return {"name": "Majority class" if task == "classification" else "Mean of target",
            "metric": metric, "cv_mean": round(float(np.mean(cv)), 4), "test_score": round(test, 4)}


def feature_importances(pipe: Pipeline) -> list[dict]:
    est = pipe[-1]
    if hasattr(est, "feature_importances_"):
        imps = np.asarray(est.feature_importances_, dtype=float)
    elif hasattr(est, "coef_"):
        coef = np.asarray(est.coef_, dtype=float)
        imps = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return []  # SVM (RBF) and KNN have no per-feature weights
    names = [str(n) for n in pipe[:-1].get_feature_names_out()]
    if len(names) != len(imps) or imps.sum() <= 0:
        return []
    imps = imps / imps.sum()
    pairs = sorted(zip(names, imps.tolist()), key=lambda kv: kv[1], reverse=True)[:10]
    return [{"feature": f, "importance": round(float(i), 4)} for f, i in pairs]


def estimator_spec(name: str, est) -> tuple[dict, str, bytes]:
    """(bundle estimator spec, weights file format, weights file bytes).
    Boosted trees are stored in their native formats (XGBoost pickles break
    across builds and OSes); scikit-learn estimators as objects, with their
    learned arrays also exported as .npz."""
    with tempfile.TemporaryDirectory() as d:
        if name == "CatBoost":
            path = os.path.join(d, "model.cbm")
            est.save_model(path, format="cbm")
            data = open(path, "rb").read()
            return {"kind": "catboost-cbm", "bytes": data}, "catboost-cbm", data
        if name in ("XGBoost", "XGBoost Leaf-wise"):
            path = os.path.join(d, "model.json")
            est.save_model(path)
            data = open(path, "rb").read()
            return {"kind": "xgboost-json", "bytes": data}, "xgboost-json", data
    arrays = {}
    inner = getattr(est, "regressor_", est)  # SVR sits inside a target-scaling wrapper
    estimators = getattr(inner, "estimators_", None) or [inner]  # one-vs-rest wraps one SVC per class
    for i, sub in enumerate(estimators):
        prefix = f"class{i}_" if len(estimators) > 1 else ""
        for attr in ("coef_", "intercept_", "support_vectors_", "dual_coef_", "_fit_X", "classes_"):
            if hasattr(sub, attr):
                arrays[prefix + attr.strip("_")] = np.asarray(getattr(sub, attr))
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return {"kind": "sklearn", "object": est}, "numpy-npz", buf.getvalue()


# -- tuning ------------------------------------------------------------------------

def cv_noise(fold_scores: list[float]) -> float:
    """Standard error of a CV mean: gains smaller than this are indistinguishable from fold luck."""
    return float(np.std(fold_scores) / np.sqrt(len(fold_scores)))


def tune(name, X, y, folds, task, device, balanced, n_classes, metric, start_score, n_trials, timeout, min_gain=0.0):
    """Optuna: STARTING_POINTS first, then multivariate TPE, which models how
    settings interact. The defaults win unless a trial beats them on CV mean.
    n_trials is a budget in full CV checks: a trial stopped after one of five
    folds uses a fifth of one. On folds of HALVING_MIN_FOLD_ROWS or more, a
    trial must be among the best third after one fold and again after three
    (successive halving); on smaller folds it stops once behind the median.
    After the starting points, the search ends once TUNING_PATIENCE checks'
    worth of folds (pruned and failed trials included) beat the best by no
    more than min_gain."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_folds = len(folds)
    starts = STARTING_POINTS[name]
    halving = min(len(va) for _, va in folds) >= HALVING_MIN_FOLD_ROWS
    best = {"params": dict(DEFAULT_PARAMS[name]), "score": start_score}
    spent = {"folds": 0, "stale": 0}
    rows = [{"trial": 0, "parameters": "Baseline Settings", "score": round(start_score, 4),
             "result": "Defaults (CV mean)", "folds": n_folds}]

    def objective(trial):
        params = search_space(name, trial)
        note = " (starting point)" if trial.number < len(starts) else ""
        row = {"trial": trial.number + 1, "parameters": json.dumps(params), "score": None}
        rows.append(row)
        try:
            value = float(np.mean(cross_validate(name, params, X, y, folds, task, device, balanced,
                                                 n_classes, metric, trial=trial)))
        except optuna.TrialPruned as e:
            row["result"] = f"Pruned {e} ({'not among the best third' if halving else 'behind the median'}){note}"
            raise
        except Exception as e:
            row["result"] = f"Failed: {str(e)[:80]}{note}"
            raise
        delta = value - best["score"]
        trial.set_user_attr("gain", delta > min_gain)
        if value > best["score"]:
            best.update(params=params, score=value)
            row["result"] = f"New best! CV +{delta:.4f}{note}"
        else:
            row["result"] = f"No improvement (CV={value:.4f}){note}"
        row["score"] = round(value, 4)
        return value

    def account(study, frozen):
        # Folds this trial used; a failed one also spent the fold it failed in.
        used = max(1, len(frozen.intermediate_values) + (frozen.state == optuna.trial.TrialState.FAIL))
        for row in rows:
            if row["trial"] == frozen.number + 1:
                row["folds"] = used
        spent["folds"] += used
        if frozen.user_attrs.get("gain"):
            spent["stale"] = 0
        elif frozen.number >= len(starts):
            spent["stale"] += used
        if spent["folds"] >= n_trials * n_folds or spent["stale"] >= TUNING_PATIENCE * n_folds:
            study.stop()

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True, n_startup_trials=len(starts)),
        pruner=(optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=3) if halving
                else optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2)),
    )
    for params in starts:
        study.enqueue_trial(params)
    # Every trial uses at least one fold, so `account` stops it before this cap.
    study.optimize(objective, n_trials=n_trials * n_folds, timeout=timeout, catch=(Exception,), callbacks=[account])
    return best["params"], best["score"], rows


# -- entry point -------------------------------------------------------------------

def _skip_reason(name: str, n_train: int) -> str | None:
    if name == "SVM" and n_train > SVM_MAX_ROWS:
        return f"skipped: kernel SVM is impractical above {SVM_MAX_ROWS:,} training rows"
    if name == "KNN" and n_train > KNN_MAX_ROWS:
        return f"skipped: KNN is impractical above {KNN_MAX_ROWS:,} training rows"
    return None


def _log(started: float, message: str) -> None:
    print(f"[{time.monotonic() - started:6.0f}s] {message}", flush=True)


def run(data: bytes, spec: dict, device: str) -> dict:
    """spec: {"task", "target", "candidates": [...], "plan": {"imbalance",
    "tuning_trials", "tuning_timeout_seconds", "time_budget_seconds",
    "input_cols", "cleaning"}}. input_cols and cleaning (both optional) are the
    raw columns a prediction takes and how runtime.prepare turns them into
    this frame's columns."""
    started = time.monotonic()
    task, target, plan = spec["task"], spec["target"], spec.get("plan", {})
    df = pd.read_parquet(io.BytesIO(data))
    X = df.drop(columns=[target])
    classification = task == "classification"
    imbalance = plan.get("imbalance") or {}
    balanced = bool(classification and imbalance.get("strategy") == "balanced_class_weights")
    metric = imbalance.get("selection_metric") or ("accuracy" if classification else "r2")
    budget = plan.get("time_budget_seconds")

    def time_left() -> float:
        return budget - (time.monotonic() - started) if budget else float("inf")

    _log(started, f"{task}, {len(df):,} rows x {X.shape[1]} columns: {', '.join(spec['candidates'])}")

    class_labels, n_classes = None, 0
    if classification:
        classes, y = np.unique(df[target].to_numpy(), return_inverse=True)
        class_labels = [python_value(c) for c in classes]
        n_classes = len(class_labels)
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
    if classification and np.bincount(y_train, minlength=n_classes).min() < 2:
        raise ValueError("After the train/test split a class has fewer than 2 training rows; collect more data.")
    folds = make_folds(task, y_train)

    candidates = []
    for name in spec["candidates"]:
        reason = _skip_reason(name, len(y_train))
        if reason is None and time_left() <= 0:
            reason = "skipped: the group's time budget ran out"
        if reason:
            candidates.append({"name": name, "status": "skipped", "reason": reason})
            _log(started, f"{name} {reason}")
            continue
        t = time.monotonic()
        try:
            cv = cross_validate(name, DEFAULT_PARAMS[name], X_train, y_train, folds, task, device, balanced,
                                n_classes, metric)
        except Exception as e:
            candidates.append({"name": name, "status": "failed", "reason": str(e)[:300]})
            _log(started, f"{name} failed after {time.monotonic() - t:.0f}s: {str(e)[:200]}")
            continue
        candidates.append({"name": name, "status": "ok", "cv_mean": round(float(np.mean(cv)), 4),
                           "cv_std": round(float(np.std(cv)), 4), "cv_folds": [round(s, 4) for s in cv]})
        _log(started, f"{name}: CV {metric} {np.mean(cv):.4f} in {time.monotonic() - t:.0f}s")

    result = {
        "task": task, "selection_metric": metric, "candidates": candidates,
        "baseline": baseline(task, y_train, y_test, folds, metric),
        "n_folds": len(folds), "test_size": int(len(y_test)),
        "imbalance_applied": {"class_weights": balanced, "selection_metric": metric},
        "library_versions": library_versions(), "device": device,
    }
    ok = [c for c in candidates if c["status"] == "ok"]
    if not ok:
        result["best"] = None
        result["seconds"] = round(time.monotonic() - started, 1)
        return result

    top = max(ok, key=lambda c: c["cv_mean"])
    best = top["name"]
    remaining = time_left()
    params, cv_mean, trials = tune(
        best, X_train, y_train, folds, task, device, balanced, n_classes, metric,
        start_score=top["cv_mean"], min_gain=cv_noise(top["cv_folds"]),
        n_trials=int(plan.get("tuning_trials", 20)) if remaining > 0 else 0,
        timeout=int(min(plan.get("tuning_timeout_seconds", 300), max(remaining, 0))),
    )
    pruned = sum(row["score"] is None for row in trials[1:])
    _log(started, f"Tuned {best}: CV {metric} {cv_mean:.4f} after {len(trials) - 1} trials ({pruned} stopped early)")

    with gpu_only(device):
        if FAMILY[best] == "gbdt":
            # Find the tree count with early stopping, then refit on the whole training split.
            _, trees = fit(best, task, params, device, balanced, n_classes, X_train, y_train)
            final, _ = fit(best, task, params, device, balanced, n_classes, X_train, y_train, n_estimators=trees)
            params = {**params, "n_estimators": trees}
        else:
            final, _ = fit(best, task, params, device, balanced, n_classes, X_train, y_train)
        test_pred = predict(final, X_test)
        train_pred = predict(final, X_train)
        proba = final.predict_proba(X_test) if classification and hasattr(final[-1], "predict_proba") else None
        decision = (final.decision_function(X_test)
                    if classification and proba is None and n_classes == 2 and hasattr(final[-1], "decision_function")
                    else None)
    _log(started, f"Fitted {best} on {len(y_train):,} rows; packaging it")

    est_spec, weights_format, weights = estimator_spec(best, final[-1])
    buf = io.BytesIO()
    joblib.dump({
        "format": BUNDLE_FORMAT, "task": task, "problem_type": task, "model_name": best,
        "feature_cols": plan.get("input_cols") or X.columns.tolist(), "model_cols": X.columns.tolist(),
        "cleaning": plan.get("cleaning"),
        "class_labels": class_labels, "preprocessor": final[0], "estimator": est_spec,
    }, buf)

    result.update({
        "best": best,
        "model_name": best,
        "cv_mean": round(float(cv_mean), 4),
        "params": params,
        "tuning_trials": trials,
        "test_metrics": full_metrics(task, y_test, test_pred, proba, decision),
        "train_metrics": full_metrics(task, y_train, train_pred),
        "train_score": score(metric, y_train, train_pred),
        "test_score": score(metric, y_test, test_pred),
        "n_classes": n_classes or None,
        "class_labels": class_labels,
        "feature_cols": plan.get("input_cols") or X.columns.tolist(),
        "model_cols": X.columns.tolist(),
        "top_features": feature_importances(final),
        "preprocessing": describe_preprocessor(final[0]),
        "y_test": y_test.tolist(),
        "y_pred": test_pred.tolist(),
        # Each test row's class probabilities (or, for a two-class SVM, its margin), for the ROC and
        # precision-recall graphs.
        "y_score": proba.tolist() if proba is not None else decision.tolist() if decision is not None else None,
        "bundle_bytes": buf.getvalue(),
        "weights_format": weights_format,
        "weights_bytes": weights,
        "seconds": round(time.monotonic() - started, 1),
    })
    _log(started, f"Done: test {metric} {result['test_score']:.4f}, model {len(result['bundle_bytes']) / 1e6:.1f} MB")
    return result
