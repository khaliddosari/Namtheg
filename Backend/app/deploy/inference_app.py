"""Shared Modal app that serves predictions for every Namtheg run.

Deployed ONCE per workspace (from Backend/, so `app` is importable):

    modal deploy app/deploy/inference_app.py

After that, the FastAPI backend just uploads each new model to the
`modelforge-models` Modal Volume and points the user at the existing endpoint
with `?run_id=<id>`. No per-run Modal deploys, no leaked apps, no image
rebuilds.

Endpoint URLs (after deploy) look like:

    https://{workspace}--modelforge-inference-predictor-predict.modal.run/?run_id=<id>
    https://{workspace}--modelforge-inference-predictor-schema.modal.run/?run_id=<id>

`{workspace}` is your Modal username — visible in any `modal app list` output.

Models load through app.training.runtime, the same code shipped in every
downloadable model package. Serving is CPU-only: models are stored in
device-independent forms, so GPU-trained weights load here unchanged.
"""
import modal

# Must equal the GPU training images (app/training/gpu_app.py) for every
# package they share, since scikit-learn objects pickled there load here, and
# the Python version must match too. tests/test_training.py enforces this.
# Update and redeploy together. xgboost-cpu is the same `xgboost` module
# without CUDA; torch comes from the CPU-only index (much smaller image).
PYTHON_VERSION = "3.12"
IMAGE_PIN = {
    "scikit-learn": "1.8.0",
    "pandas": "2.3.3",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "joblib": "1.5.3",
    "xgboost-cpu": "3.4.1",
    "catboost": "1.2.10",
    "timm": "1.0.30",
    "sentence-transformers": "6.1.0",
    "chronos-forecasting": "2.3.2",
}
TORCH_PIN = {"torch": "2.14.0", "torchvision": "0.29.0"}

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*(f"{name}=={version}" for name, version in TORCH_PIN.items()),
                 index_url="https://download.pytorch.org/whl/cpu")
    .pip_install(*(f"{name}=={version}" for name, version in IMAGE_PIN.items()), "pillow", "fastapi[standard]")
    .add_local_python_source("app")
)

# All run models live here; the backend writes /{run_id}/model.joblib into it
# before pointing the user at the predict endpoint.
models_volume = modal.Volume.from_name("modelforge-models", create_if_missing=True)

app = modal.App("modelforge-inference", image=image)


@app.cls(
    volumes={"/models": models_volume},
    # Scale-to-zero when no traffic, but keep warm for 5 min between requests so
    # a quick demo doesn't pay the cold-start cost twice.
    min_containers=0,
    scaledown_window=300,
    memory=4096,
)
class Predictor:
    """Loads + caches per-run model bundles on demand from the shared volume."""

    # Keep this small to bound container memory. Older models get evicted.
    _CACHE_LIMIT = 10

    @modal.enter()
    def init(self):
        # Insertion-ordered dict acts as a simple LRU.
        self._cache: dict[str, dict] = {}

    def _bundle(self, run_id: str) -> dict:
        # Reject anything that isn't a plain alphanumeric/hex run id — defends
        # against path traversal via `..` etc.
        if not run_id or not all(c.isalnum() or c in "-_" for c in run_id):
            raise ValueError("Invalid run_id.")

        if run_id in self._cache:
            # Touch for LRU: move to end.
            bundle = self._cache.pop(run_id)
            self._cache[run_id] = bundle
            return bundle

        import joblib

        from app.training.runtime import load_bundle

        # The backend writes models as /models/<run_id>/model.joblib, and calls
        # vol.commit() so the file is visible here. We also call reload() to be
        # sure our container's view is fresh — cheap on a hot path.
        models_volume.reload()
        bundle = load_bundle(joblib.load(f"/models/{run_id}/model.joblib"))

        # Evict oldest if at capacity.
        if len(self._cache) >= self._CACHE_LIMIT:
            self._cache.pop(next(iter(self._cache)))
        self._cache[run_id] = bundle
        return bundle

    @modal.fastapi_endpoint(method="POST", docs=True)
    def predict(self, run_id: str, payload: dict):
        """POST /?run_id=<id>. The body depends on the model's task:
        - classification / regression / clustering: {"features": {...}} or {"rows": [[...]]}
        - forecasting: {} to forecast from the end of the training data, or
          {"history": [{"timestamp", "value", "series_id"?}, ...]}
        - image classification: {"images": ["<base64>", ...]}
        """
        import base64

        import numpy as np
        import pandas as pd

        try:
            bundle = self._bundle(run_id)
        except FileNotFoundError:
            return {"error": f"Model for run_id={run_id} not found in volume."}
        except ValueError as e:
            return {"error": str(e)}

        model = bundle["model"]
        task = bundle.get("task") or bundle.get("problem_type")
        model_name = bundle.get("model_name", "model")
        class_labels = bundle.get("class_labels")

        try:
            if task == "forecasting":
                history = pd.DataFrame(payload["history"]) if payload.get("history") else None
                forecast = model.forecast(history)
                forecast["timestamp"] = forecast["timestamp"].astype(str)
                return {"forecast": forecast.to_dict(orient="records"), "model": model_name}

            if task == "image_classification":
                images = [base64.b64decode(img) for img in payload.get("images") or []]
                if not images:
                    return {"error": "Provide 'images': a list of base64-encoded image files."}
                probs = model.predict_proba(images)
                preds = probs.argmax(axis=1).tolist()
                return {"predictions": preds, "predicted_labels": [class_labels[p] for p in preds],
                        "probabilities": probs.tolist(), "class_labels": class_labels, "model": model_name}
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

        feature_cols = bundle["feature_cols"]
        if "features" in payload and isinstance(payload["features"], dict):
            row = {c: payload["features"].get(c) for c in feature_cols}
            df = pd.DataFrame([row], columns=feature_cols)
        elif "rows" in payload and isinstance(payload["rows"], list):
            try:
                df = pd.DataFrame(payload["rows"], columns=feature_cols)
            except Exception as e:
                return {
                    "error": (
                        f"Each row must have {len(feature_cols)} values in the "
                        f"order returned by /schema. {e}"
                    )
                }
        else:
            return {
                "error": (
                    "Provide either 'features' (object of column->value) or "
                    "'rows' (array of arrays). Call /schema for the expected columns."
                )
            }

        try:
            # ravel: CatBoost multiclass predict returns shape (n, 1).
            preds_list = np.asarray(model.predict(df)).ravel().tolist()
            if task == "clustering":
                return {"clusters": preds_list, "model": model_name,
                        "note": "-1 means the row is not close to any cluster (outlier)."}
            result = {"predictions": preds_list, "model": model_name}

            if task == "classification" or bundle.get("problem_type") == "classification":
                if class_labels:
                    result["predicted_labels"] = [
                        class_labels[int(p)] if 0 <= int(p) < len(class_labels) else None
                        for p in preds_list
                    ]
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(df)
                    result["probabilities"] = np.asarray(probs).tolist()
                    if class_labels:
                        result["class_labels"] = list(class_labels)
            return result
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

    @modal.fastapi_endpoint(method="GET", docs=True)
    def schema(self, run_id: str):
        """GET /?run_id=<id> — what this model expects and example payloads."""
        try:
            bundle = self._bundle(run_id)
        except FileNotFoundError:
            return {"error": f"Model for run_id={run_id} not found in volume."}
        except ValueError as e:
            return {"error": str(e)}

        task = bundle.get("task") or bundle.get("problem_type")
        feature_cols = bundle.get("feature_cols") or []
        class_labels = bundle.get("class_labels")
        out = {
            "run_id": run_id,
            "task": task,
            "model_name": bundle.get("model_name", "model"),
            "problem_type": bundle.get("problem_type"),
            "feature_cols": feature_cols,
            "class_labels": list(class_labels) if class_labels else None,
        }
        if task == "forecasting":
            cfg = bundle["forecast"]
            out.update(horizon=cfg["horizon"], freq=cfg["freq"], example={"history": [
                {"timestamp": "2026-01-01", "value": 0.0, "series_id": next(iter(cfg["history"]))}]})
        elif task == "image_classification":
            out["example"] = {"images": ["<base64-encoded image file>"]}
        else:
            out.update(example_features={"features": {c: 0 for c in feature_cols}},
                       example_rows={"rows": [[0 for _ in feature_cols]]})
        return out
