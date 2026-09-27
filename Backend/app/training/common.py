"""Helpers shared by every training engine. Third-party imports only: this
package runs inside the GPU containers without the rest of `app`."""
import platform
from contextlib import contextmanager

import numpy as np
import pandas as pd
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
from sklearn.model_selection import KFold, StratifiedKFold

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FOLDS = 5
MIN_ROWS_PER_CLASS = 3

# Metrics where smaller is better; everything else is maximised.
LOWER_IS_BETTER = {"mase", "smape", "rmse", "mae", "davies_bouldin"}


class CPUFallback(RuntimeError):
    """An accelerated estimator ran on CPU instead of the GPU."""


@contextmanager
def gpu_only(device: str):
    """On CUDA, fail if any cuML-accelerated scikit-learn call fell back to CPU.

    cuml.accel silently runs unsupported estimator/parameter combinations on
    CPU; its profiler records every such call with the reason, so the
    candidate is rejected instead of quietly training on CPU. A no-op off-GPU
    (tests run the same code on CPU as a harness).
    """
    if not device.startswith("cuda"):
        yield
        return
    import cuml.accel

    with cuml.accel.profile(quiet=True) as prof:
        yield
    fallbacks = {name: sorted(stats.fallback_reasons) or ["no reason given"]
                 for name, stats in prof.method_calls.items() if stats.cpu_calls}
    if fallbacks:
        detail = "; ".join(f"{name}: {', '.join(reasons)}" for name, reasons in fallbacks.items())
        raise CPUFallback(f"Refused: ran on CPU instead of the GPU ({detail}).")


def higher_is_better(metric: str) -> bool:
    return metric not in LOWER_IS_BETTER


def score(metric: str, y_true, y_pred) -> float:
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if metric == "r2":
        return float(r2_score(y_true, y_pred))
    raise ValueError(f"Unknown metric {metric!r}")


def full_metrics(problem_type: str, y_true, y_pred, proba=None, decision=None) -> dict:
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
    if len(np.unique(y_true)) == 2:
        # Ranking metrics need a score per row: probabilities, or an SVM's margin.
        ranked = proba[:, 1] if proba is not None and proba.shape[1] == 2 else decision
        if ranked is not None:
            out["roc_auc"] = float(roc_auc_score(y_true, ranked))
            out["pr_auc"] = float(average_precision_score(y_true, ranked))
    return out


def make_folds(problem_type: str, y_train: np.ndarray) -> list:
    if problem_type == "classification":
        n = max(2, min(MAX_FOLDS, int(pd.Series(y_train).value_counts().min())))
        splitter = StratifiedKFold(n_splits=n, shuffle=True, random_state=RANDOM_STATE)
    else:
        n = max(2, min(MAX_FOLDS, len(y_train) // 10))
        splitter = KFold(n_splits=n, shuffle=True, random_state=RANDOM_STATE)
    return list(splitter.split(np.zeros(len(y_train)), y_train))


def python_value(v):
    """numpy scalars -> plain Python, for results that cross process boundaries."""
    return v.item() if hasattr(v, "item") else v


def library_versions() -> dict:
    """Exact versions the model was trained with; loading it elsewhere needs the same ones."""
    import joblib
    import sklearn

    versions = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__, "joblib": joblib.__version__}
    for name in ("xgboost", "catboost", "torch", "timm", "sentence_transformers", "chronos"):
        try:
            module = __import__(name)
            versions[name.replace("_", "-")] = getattr(module, "__version__", "installed")
        except ImportError:
            pass
    return versions
