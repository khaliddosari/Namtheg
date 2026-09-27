"""Backend side of training.

Every model is fit on the GPU service (app/training/gpu_app.py). This module
ships the engineered dataset there and stores what comes back. There is
deliberately no local or CPU fallback: if the GPU service is unavailable, the
run fails with that error rather than quietly training somewhere else.
"""
import logging

import numpy as np

from app import storage
from app.config import settings

log = logging.getLogger(__name__)

GPU_APP_NAME = "namtheg-train-gpu"
WEIGHTS_FILES = {"xgboost-json": "model_weights.json", "catboost-cbm": "model_weights.cbm"}


def _remote_train(data: bytes, target: str, problem_type: str, plan: dict) -> dict:
    import modal

    try:
        return modal.Function.from_name(GPU_APP_NAME, "train").remote(data, target, problem_type, plan)
    except modal.exception.NotFoundError as e:
        raise RuntimeError(
            f"The GPU training service '{GPU_APP_NAME}' is not deployed. Run: modal deploy app/training/gpu_app.py"
        ) from e


def train_model(run_id: str, target: str, problem_type: str, imbalance_plan: dict) -> dict:
    """Train on the GPU and persist every artifact. Returns metrics in the
    shape result.json has always used ({"model_name", "score", "score_metric",
    "extra"})."""
    plan = {
        "imbalance": imbalance_plan,
        "tuning_trials": settings.tuning_trials,
        "tuning_timeout_seconds": settings.tuning_timeout_seconds,
    }
    log.info("Sending run %s to the GPU training service.", run_id)
    out = _remote_train(storage.engineered_path(run_id).read_bytes(), target, problem_type, plan)
    log.info("Run %s trained on %s in %ss.", run_id, out.get("hardware", {}).get("gpu"), out.get("seconds"))
    return _store(run_id, target, out)


def _save_bytes(run_id: str, name: str, data: bytes) -> None:
    (storage.run_dir(run_id) / name).write_bytes(data)
    storage.persist(run_id, name)


def _save_array(run_id: str, name: str, values) -> None:
    np.save(storage.run_dir(run_id) / name, values, allow_pickle=True)
    storage.persist(run_id, name)


def _store(run_id: str, target: str, out: dict) -> dict:
    labels = out["class_labels"]
    y_test, y_pred = np.asarray(out["y_test"]), np.asarray(out["y_pred"])
    if labels:
        # Plot real class names, not the integer codes the models trained on.
        names = np.asarray([str(label) for label in labels], dtype=object)
        y_test, y_pred = names[y_test.astype(int)], names[y_pred.astype(int)]
    _save_array(run_id, "y_test.npy", y_test)
    _save_array(run_id, "y_pred.npy", y_pred)

    _save_bytes(run_id, "model.joblib", out["bundle_bytes"])
    weights_file = WEIGHTS_FILES[out["weights_format"]]
    _save_bytes(run_id, weights_file, out["weights_bytes"])

    head = storage.engineered_head(run_id, 1)
    storage.write_json(run_id, "model_meta.json", {
        "model_name": out["model_name"],
        "problem_type": out["problem_type"],
        "target": target,
        "feature_cols": out["feature_cols"],
        "feature_dtypes": {c: str(head[c].dtype) for c in out["feature_cols"] if c in head.columns},
        "class_labels": labels,
        "params": out["params"],
        "preprocessing": out["preprocessing"],
        "weights_file": weights_file,
        "weights_format": out["weights_format"],
        "library_versions": out["library_versions"],
        "hardware": out.get("hardware"),
    })

    metric = out["selection_metric"]
    test, train = out["test_metrics"], out["train_metrics"]
    extra = {
        "cv_mean": out["cv_mean"],
        "train_score": out["train_score"],
        "test_score": out["test_score"],
        "overfit_gap": round(out["train_score"] - out["cv_mean"], 4),
        "test_metrics": test,
        "train_metrics": train,
        "n_folds": out["n_folds"],
        "test_size": out["test_size"],
        "params": out["params"],
        "all_models": out["all_models"],
        "top_features": out["top_features"],
        "tuning_trials": out["tuning_trials"],
        "baseline": out["baseline"],
        "imbalance_applied": out["imbalance_applied"],
        "hardware": out.get("hardware"),
        "training_seconds": out.get("seconds"),
    }
    # Keys the current frontend reads.
    if out["problem_type"] == "classification":
        extra.update({
            "train_accuracy": train["accuracy"],
            "f1_macro": test["f1_macro"],
            "balanced_accuracy": test["balanced_accuracy"],
            "n_classes": out["n_classes"],
        })
        if metric == "accuracy":
            extra["cv_accuracy_mean"] = out["cv_mean"]
    else:
        extra.update({"train_r2": train["r2"], "cv_r2_mean": out["cv_mean"], "rmse": test["rmse"], "mae": test["mae"]})

    metrics = {"model_name": out["model_name"], "score": out["cv_mean"], "score_metric": metric, "extra": extra}
    storage.write_json(run_id, "metrics.json", metrics)
    return metrics
