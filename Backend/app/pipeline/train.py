"""Backend side of training.

Every model is fit on the GPU service (app/training/gpu_app.py). This module
fans work out to it in parallel (one GPU container per model group or
candidate), then stores what comes back. There is deliberately no local or
CPU fallback: if the GPU service is unavailable, the run fails with that
error rather than quietly training somewhere else.

`_gpu_map` and `_gpu_call` are the only places that talk to the GPU service.
"""
import io
import logging
import zipfile

import numpy as np
import pandas as pd

from app import storage
from app.config import settings
from app.training import clustering, forecasting, tabular, text, vision
from app.training.common import higher_is_better, score
from app.training.runtime import attach_text_features

log = logging.getLogger(__name__)

GPU_APP_NAME = "namtheg-train-gpu"
TRAINING_GPU = "H200"  # must equal app/training/gpu_app.GPU (tests/test_training.py checks)
WEIGHTS_FILES = {"xgboost-json": "model_weights.json", "catboost-cbm": "model_weights.cbm",
                 "numpy-npz": "model_weights.npz"}
ESTIMATOR_KIND = {"xgboost-json": "xgboost-json", "catboost-cbm": "catboost-cbm", "numpy-npz": "sklearn"}
FORECAST_KIND = {"Chronos-2": "chronos", "XGBoost (lags)": "xgboost-json"}  # the rest are torch nets


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


def _gpu_call(function: str, *args):
    import modal

    try:
        return _function(function).remote(*args)
    except modal.exception.NotFoundError as e:
        raise _not_deployed(e) from e


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


# -- text ---------------------------------------------------------------------------

def embed_text(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict | None, dict]:
    """Replace free-text columns with pretrained-encoder features (computed on
    the GPU). Returns (frame, text spec for the model bundle, report)."""
    if not columns:
        return frame, None, {}
    payload = {c: [None if pd.isna(v) else str(v) for v in frame[c]] for c in columns}
    out = _gpu_call("embed_text", payload, text.ENCODER)
    for c in columns:
        Z = np.frombuffer(out["features"][c], dtype=np.float32).reshape(len(frame), -1)
        frame = attach_text_features(frame, c, Z)
    spec = {"encoder": out["encoder"],
            "columns": {c: {"mean": p["mean"], "components": p["components"]} for c, p in out["columns"].items()}}
    report = {c: {"dimensions": len(p["components"]), "explained_variance": p["explained_variance"]}
              for c, p in out["columns"].items()}
    return frame, spec, {"encoder": out["encoder"], "columns": report}


def _training_frame(run_id: str, target: str | None, text_columns: list[str]):
    engineered = storage.load_engineered(run_id)
    raw_cols = [c for c in engineered.columns if c != target]
    frame, text_spec, text_report = embed_text(engineered, text_columns)
    return frame, raw_cols, text_spec, text_report


# -- tabular -------------------------------------------------------------------------

def train_tabular(run_id: str, target: str, task: str, imbalance_plan: dict, text_columns: list[str]) -> dict:
    """Model groups (boosted trees; SVM/KNN/linear) train in parallel on
    separate GPUs; each tunes its own best candidate; the best group wins on CV."""
    frame, raw_cols, text_spec, text_report = _training_frame(run_id, target, text_columns)
    plan = {"imbalance": imbalance_plan, "tuning_trials": settings.tuning_trials,
            "tuning_timeout_seconds": settings.tuning_timeout_seconds, "text": text_spec,
            "raw_feature_cols": raw_cols}
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
    head = storage.engineered_head(run_id, 1)
    storage.write_json(run_id, "model_meta.json", {
        "task": task, "model_name": _display(best["model_name"], task), "problem_type": task, "target": target,
        "feature_cols": best["feature_cols"],
        "feature_dtypes": {c: str(head[c].dtype) for c in best["feature_cols"] if c in head.columns},
        "class_labels": best["class_labels"], "params": best["params"], "preprocessing": best["preprocessing"],
        "text": text_report or None, "weights_file": weights_file, "weights_format": best["weights_format"],
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
        "text_features": text_report or None,
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


# -- clustering ----------------------------------------------------------------------------

def train_clustering(run_id: str, text_columns: list[str]) -> dict:
    frame, raw_cols, text_spec, text_report = _training_frame(run_id, None, text_columns)
    spec = {"task": "clustering", "candidates": list(clustering.CANDIDATES),
            "plan": {"text": text_spec, "raw_feature_cols": raw_cols}}
    ok, errors = _split_results(_gpu_map("train_classic", _frame_bytes(frame), [spec], run_id), "Clustering")
    if not ok:
        raise RuntimeError("Clustering failed on the GPU service: " + " | ".join(errors))
    out = ok[0]
    if not out.get("best"):
        reasons = [f"{c['name']}: {c.get('reason') or c['status']}" for c in out["candidates"]]
        raise RuntimeError("No clustering method found structure (2+ clusters, at most half noise). " +
                           " | ".join(reasons))

    _save_bytes(run_id, "model.joblib", out["bundle_bytes"])
    _save_array(run_id, "cluster_labels.npy", np.asarray(out["labels"]))
    storage.write_json(run_id, "projection.json", out["projection"])
    storage.write_json(run_id, "model_meta.json", {
        "task": "clustering", "model_name": out["model_name"], "problem_type": "clustering", "target": None,
        "feature_cols": out["feature_cols"], "class_labels": None, "params": out["params"],
        "estimator_kind": "cluster-assignment",
        "text": text_report or None, "library_versions": out["library_versions"], "hardware": out.get("hardware"),
    })
    m = out["metrics"]
    extra = {
        "cv_mean": m["silhouette"],
        "cluster_metrics": m,
        "all_models": sorted([{"name": c["name"], "cv_mean": c["cv_mean"], "cv_std": 0.0,
                               "n_clusters": c["n_clusters"], "noise_share": c["noise_share"]}
                              for c in out["candidates"] if c["status"] == "ok"],
                             key=lambda c: c["cv_mean"], reverse=True),
        "excluded_models": [{"name": c["name"], "status": c["status"], "reason": c.get("reason")}
                            for c in out["candidates"] if c["status"] != "ok"],
        "kmeans_curve": out["kmeans_curve"],
        "cluster_profiles": cluster_profiles(storage.load_engineered(run_id), np.asarray(out["labels"])),
        "params": out["params"],
        "text_features": text_report or None,
        "hardware": out.get("hardware"),
        "training_seconds": out["seconds"],
    }
    metrics = {"model_name": out["model_name"], "score": m["silhouette"], "score_metric": "silhouette",
               "higher_is_better": True, "extra": extra}
    storage.write_json(run_id, "metrics.json", metrics)
    return metrics


def cluster_profiles(frame: pd.DataFrame, labels: np.ndarray, max_columns: int = 12) -> list[dict]:
    """Per cluster: size, and how its numeric means and top categories differ."""
    frame = frame.assign(_cluster=labels)
    numeric = [c for c in frame.select_dtypes("number").columns if c != "_cluster"][:max_columns]
    categorical = [c for c in frame.columns if c not in numeric and c != "_cluster"][:max_columns]
    profiles = []
    for cid, g in frame.groupby("_cluster"):
        profiles.append({
            "cluster": int(cid),
            "size": int(len(g)),
            "share": round(len(g) / len(frame), 4),
            "means": {c: round(float(g[c].mean()), 4) for c in numeric if g[c].notna().any()},
            "top_values": {c: str(g[c].mode().iloc[0]) for c in categorical if g[c].notna().any()},
        })
    return profiles


# -- forecasting ----------------------------------------------------------------------------

def train_forecasting(run_id: str, ts: dict) -> dict:
    """Every forecaster runs on its own GPU container, all at once."""
    specs = [{"task": "forecasting", "candidate": name, "horizon": ts["horizon"], "freq": ts["freq"],
              "season": ts["season"], "input_length": ts["input_length"], "windows": ts["windows"]}
             for name in forecasting.CANDIDATES]
    results = _gpu_map("train_deep", storage.engineered_path(run_id).read_bytes(), specs, run_id)
    ok, errors = [], []
    for spec, r in zip(specs, results):
        if isinstance(r, Exception):
            errors.append({"name": spec["candidate"], "status": "failed", "reason": str(r)[:300]})
        else:
            ok.append(r)
    if not ok:
        raise RuntimeError("No forecaster trained successfully. " + " | ".join(e["reason"] for e in errors))
    best = min(ok, key=lambda r: r["metrics"]["mase"])

    _save_bytes(run_id, "model.joblib", best["bundle_bytes"])
    forecast = pd.DataFrame(best["forecast"])
    forecast.to_csv(storage.run_dir(run_id) / "forecast.csv", index=False, encoding="utf-8-sig")
    storage.persist(run_id, "forecast.csv")
    storage.write_json(run_id, "backtest.json", {"backtest": best["backtest"], "forecast": best["forecast"][:500]})
    storage.write_json(run_id, "model_meta.json", {
        "task": "forecasting", "model_name": best["candidate"], "problem_type": "forecasting",
        "target": ts["target"], "feature_cols": [], "class_labels": None,
        "estimator_kind": FORECAST_KIND.get(best["candidate"], "torch"),
        "forecast": {k: ts[k] for k in ("freq", "season", "horizon", "input_length", "windows")},
        "library_versions": best["library_versions"], "hardware": best.get("hardware"),
    })
    seasonal = ts["baselines"].get("Seasonal naive", {})
    extra = {
        "cv_mean": best["metrics"]["mase"],
        "backtest_metrics": best["metrics"],
        "all_models": sorted([{"name": r["candidate"], "cv_mean": r["metrics"]["mase"], "cv_std": 0.0,
                               "smape": r["metrics"]["smape"]} for r in ok], key=lambda m: m["cv_mean"]),
        "excluded_models": errors,
        "baseline": {"name": "Seasonal naive", "metric": "mase", "cv_mean": seasonal.get("mase"),
                     **{k: v for k, v in seasonal.items() if k != "mase"}},
        "baselines": ts["baselines"],
        "forecast_rows": len(forecast),
        "timeseries": {k: ts[k] for k in ("freq", "season", "horizon", "input_length", "windows", "n_series",
                                            "start", "end", "interpolated_steps")},
        "hardware": best.get("hardware"),
        "training_seconds": max(r["seconds"] for r in ok),
    }
    metrics = {"model_name": best["candidate"], "score": best["metrics"]["mase"], "score_metric": "mase",
               "higher_is_better": False, "extra": extra}
    storage.write_json(run_id, "metrics.json", metrics)
    return metrics


# -- images -----------------------------------------------------------------------------------

def image_archive(run_id: str) -> bytes:
    """The manifest plus every image it references, as one zip for the GPU.
    Images aren't mirrored to R2 one by one; if they're missing locally they
    are re-extracted from the raw upload (extraction is deterministic)."""
    manifest = storage.load_dataset(run_id)
    root = storage.run_dir(run_id)
    if not (root / manifest["image"].iloc[0]).exists():
        from app.data.images import ingest_image_zip

        ingest_image_zip(storage.artifact_path(run_id, "raw/source.zip"), root)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED, strict_timestamps=False) as z:
        z.writestr("dataset.parquet", _frame_bytes(manifest[["image", "label"]]))
        for rel in manifest["image"]:
            z.write(root / rel, rel)
    return buf.getvalue()


def train_images(run_id: str, imbalance_plan: dict) -> dict:
    """Each pretrained CNN fine-tunes on its own GPU container, all at once."""
    plan = {"imbalance": imbalance_plan}
    specs = [{"task": "image_classification", "candidate": name, "plan": plan} for name in vision.CANDIDATES]
    results = _gpu_map("train_deep", image_archive(run_id), specs, run_id)
    ok, errors = [], []
    for spec, r in zip(specs, results):
        if isinstance(r, Exception):
            errors.append({"name": spec["candidate"], "status": "failed", "reason": str(r)[:300]})
        else:
            ok.append(r)
    if not ok:
        raise RuntimeError("No image model trained successfully. " + " | ".join(e["reason"] for e in errors))
    best = max(ok, key=lambda r: r["val_score"])
    metric = best["selection_metric"]

    _save_labels(run_id, best["y_test"], best["y_pred"], best["class_labels"])
    _save_bytes(run_id, "model.joblib", best["bundle_bytes"])
    storage.write_json(run_id, "model_meta.json", {
        "task": "image_classification", "model_name": best["candidate"], "problem_type": "classification",
        "target": "label", "feature_cols": [], "class_labels": best["class_labels"], "estimator_kind": "timm",
        "library_versions": best["library_versions"], "hardware": best.get("hardware"),
    })

    # Feature-blind baseline on the same test split: always predict the
    # training split's most common class.
    labels = storage.load_dataset(run_id)["label"].astype(str).to_numpy()
    classes, y = np.unique(labels, return_inverse=True)
    train_idx, _, test_idx = vision.split(y)
    majority = np.bincount(y[train_idx]).argmax()
    base = score(metric, y[test_idx], np.full(len(test_idx), majority))
    test = best["test_metrics"]
    extra = {
        "cv_mean": best["val_score"],
        "test_score": best["test_score"],
        "test_metrics": test,
        "epochs": best["epochs"],
        "training_curve": best["history"],
        "split_sizes": best["split_sizes"],
        "all_models": sorted([{"name": r["candidate"], "cv_mean": r["val_score"], "cv_std": 0.0}
                              for r in ok], key=lambda m: m["cv_mean"], reverse=True),
        "excluded_models": errors,
        "baseline": {"name": "Majority class", "metric": metric, "cv_mean": round(base, 4),
                     "test_score": round(base, 4)},
        "imbalance_applied": best["imbalance_applied"],
        "f1_macro": test["f1_macro"], "balanced_accuracy": test["balanced_accuracy"],
        "n_classes": len(best["class_labels"]),
        "hardware": best.get("hardware"),
        "training_seconds": max(r["seconds"] for r in ok),
    }
    metrics = {"model_name": best["candidate"], "score": best["val_score"], "score_metric": metric,
               "higher_is_better": higher_is_better(metric), "extra": extra}
    storage.write_json(run_id, "metrics.json", metrics)
    return metrics
