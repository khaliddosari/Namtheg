"""Backend side of training.

Every model is fit on the GPU service (app/training/gpu_app.py). This module
fans work out to it in parallel (one GPU container per model group), then
stores what comes back. There is deliberately no local or CPU fallback: if
the GPU service is unavailable, the run fails with that error rather than
quietly training somewhere else.

`_gpu_map` is the only place that talks to the GPU service.
"""
import io
import logging

import numpy as np
import pandas as pd

from app import storage
from app.config import settings
from app.training import tabular

log = logging.getLogger(__name__)

GPU_APP_NAME = "namtheg-train-gpu"
TRAINING_GPU = "H200"  # must equal app/training/gpu_app.GPU (tests/test_training.py checks)
WEIGHTS_FILES = {"xgboost-json": "model_weights.json", "catboost-cbm": "model_weights.cbm",
                 "numpy-npz": "model_weights.npz"}
ESTIMATOR_KIND = {"xgboost-json": "xgboost-json", "catboost-cbm": "catboost-cbm", "numpy-npz": "sklearn"}


class GPUServiceError(RuntimeError):
    pass


def _function(name: str):
    import modal

    return modal.Function.from_name(GPU_APP_NAME, name)


def _not_deployed(e: Exception) -> GPUServiceError:
    return GPUServiceError(
        f"The GPU training service '{GPU_APP_NAME}' is not deployed ({e}). Run: modal deploy app/training/gpu_app.py"
    )


def _gpu_map(function: str, data: bytes, specs: list[dict], run_id: str) -> list:
    """Run `function` once per spec, all at once on separate GPU containers.
    Returns each call's result, or the exception it raised."""
    import modal

    try:
        fn = _function(function)
        calls = [fn.spawn(data, spec) for spec in specs]
    except modal.exception.NotFoundError as e:
        raise _not_deployed(e) from e
    # Recorded so that cancelling the run also stops its GPUs.
    storage.update_status(run_id, gpu_calls=[c.object_id for c in calls])
    results = []
    for call in calls:
        try:
            results.append(call.get())
        except Exception as e:
            results.append(e)
    return results


# -- shared storage helpers ----------------------------------------------------------

def _save_bytes(run_id: str, name: str, data: bytes) -> None:
    (storage.run_dir(run_id) / name).write_bytes(data)
    storage.persist(run_id, name)


def _save_array(run_id: str, name: str, values) -> None:
    np.save(storage.run_dir(run_id) / name, values, allow_pickle=True)
    storage.persist(run_id, name)


def _frame_bytes(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


def _save_labels(run_id: str, y_test, y_pred, labels) -> None:
    y_test, y_pred = np.asarray(y_test), np.asarray(y_pred)
    if labels:
        # Plot real class names, not the integer codes the models trained on.
        names = np.asarray([str(label) for label in labels], dtype=object)
        y_test, y_pred = names[y_test.astype(int)], names[y_pred.astype(int)]
    _save_array(run_id, "y_test.npy", y_test)
    _save_array(run_id, "y_pred.npy", y_pred)


def _split_results(results: list, what: str) -> tuple[list[dict], list[str]]:
    ok, errors = [], []
    for r in results:
        if isinstance(r, Exception):
            log.warning("%s failed on the GPU service: %s", what, r)
            errors.append(str(r)[:300])
        else:
            ok.append(r)
    return ok, errors


def _display(name: str, task: str) -> str:
    if name == "Linear":
        return "Logistic Regression" if task == "classification" else "Ridge Regression"
    return name


# -- tabular -------------------------------------------------------------------------

def train_tabular(run_id: str, target: str, task: str, imbalance_plan: dict) -> dict:
    """Model groups (boosted trees; SVM/KNN/linear) train in parallel on
    separate GPUs; each tunes its own best candidate; the best group wins on CV."""
    frame = storage.load_engineered(run_id)
    fe = storage.read_json(run_id, "feature_engineering.json") or {}
    plan = {"imbalance": imbalance_plan, "tuning_trials": settings.tuning_trials,
            "tuning_timeout_seconds": settings.tuning_timeout_seconds,
            # The model takes the raw columns and cleans them itself (runtime.prepare).
            "input_cols": fe.get("input_columns"), "cleaning": fe.get("cleaning")}
    specs = [{"task": task, "target": target, "candidates": list(group), "plan": plan} for group in tabular.GROUPS]
    ok, errors = _split_results(_gpu_map("train_classic", _frame_bytes(frame), specs, run_id), "Model group")

    candidates = [c for r in ok for c in r["candidates"]]
    winners = [r for r in ok if r.get("best")]
    if not winners:
        reasons = [f"{c['name']}: {c.get('reason')}" for c in candidates if c["status"] != "ok"] + errors
        raise RuntimeError("No model trained successfully. " + " | ".join(reasons))
    best = max(winners, key=lambda r: r["cv_mean"])

    _save_labels(run_id, best["y_test"], best["y_pred"], best["class_labels"])
    _save_bytes(run_id, "model.joblib", best["bundle_bytes"])
    weights_file = WEIGHTS_FILES[best["weights_format"]]
    _save_bytes(run_id, weights_file, best["weights_bytes"])
    head = storage.dataset_head(run_id, 1)  # predictions take the raw columns
    storage.write_json(run_id, "model_meta.json", {
        "task": task, "model_name": _display(best["model_name"], task), "problem_type": task, "target": target,
        "feature_cols": best["feature_cols"], "model_cols": best.get("model_cols"), "cleaning": fe.get("cleaning"),
        "feature_dtypes": {c: str(head[c].dtype) for c in best["feature_cols"] if c in head.columns},
        "class_labels": best["class_labels"], "params": best["params"], "preprocessing": best["preprocessing"],
        "weights_file": weights_file, "weights_format": best["weights_format"],
        "estimator_kind": ESTIMATOR_KIND[best["weights_format"]],
        "library_versions": best["library_versions"], "hardware": best.get("hardware"),
    })

    metric = best["selection_metric"]
    test, train = best["test_metrics"], best["train_metrics"]
    all_models = [{"name": _display(c["name"], task), "cv_mean": c["cv_mean"], "cv_std": c["cv_std"]}
                  for c in candidates if c["status"] == "ok"]
    extra = {
        "cv_mean": best["cv_mean"],
        "train_score": best["train_score"],
        "test_score": best["test_score"],
        "overfit_gap": round(best["train_score"] - best["cv_mean"], 4),
        "test_metrics": test,
        "train_metrics": train,
        "n_folds": best["n_folds"],
        "test_size": best["test_size"],
        "params": best["params"],
        "all_models": sorted(all_models, key=lambda m: m["cv_mean"], reverse=True),
        "excluded_models": [{"name": _display(c["name"], task), "status": c["status"], "reason": c.get("reason")}
                            for c in candidates if c["status"] != "ok"] + [{"name": "model group", "status": "failed",
                                                                            "reason": e} for e in errors],
        "top_features": best["top_features"],
        "tuning_trials": best["tuning_trials"],
        "baseline": best["baseline"],
        "imbalance_applied": best["imbalance_applied"],
        "hardware": best.get("hardware"),
        "training_seconds": max(r["seconds"] for r in ok),
    }
    # Keys the current frontend reads.
    if task == "classification":
        extra.update({"train_accuracy": train["accuracy"], "f1_macro": test["f1_macro"],
                      "balanced_accuracy": test["balanced_accuracy"], "n_classes": best["n_classes"]})
        if metric == "accuracy":
            extra["cv_accuracy_mean"] = best["cv_mean"]
    else:
        extra.update({"train_r2": train["r2"], "cv_r2_mean": best["cv_mean"], "rmse": test["rmse"], "mae": test["mae"]})

    metrics = {"model_name": _display(best["model_name"], task), "score": best["cv_mean"], "score_metric": metric,
               "higher_is_better": True, "extra": extra}
    storage.write_json(run_id, "metrics.json", metrics)
    return metrics
