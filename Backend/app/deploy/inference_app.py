"""Shared Modal app that serves predictions for every Namtheg run.

Deployed ONCE per workspace:

    modal deploy app/deploy/inference_app.py

After that, the FastAPI backend just uploads each new model to the
`modelforge-models` Modal Volume and points the user at the existing endpoint
with `?run_id=<id>`. No per-run Modal deploys, no leaked apps, no image
rebuilds.

Endpoint URLs (after deploy) look like:

    https://{workspace}--modelforge-inference-predictor-predict.modal.run/?run_id=<id>
    https://{workspace}--modelforge-inference-predictor-schema.modal.run/?run_id=<id>

`{workspace}` is your Modal username — visible in any `modal app list` output.

Bump the pinned library versions in `IMAGE_PIN` whenever the backend's
training environment changes, then `modal deploy` again to refresh the image.
"""
import modal


# Must equal the GPU training image (app/training/gpu_app.py) for every
# package they share, so the pipelines pickled there unpickle here; the Python
# version must match too. tests/test_training.py enforces this. Update both and
# redeploy both together. xgboost-cpu is the same `xgboost` module without CUDA:
# serving runs on CPU, and trained models are switched to device="cpu".
PYTHON_VERSION = "3.12"
IMAGE_PIN = {
    "scikit-learn": "1.8.0",
    "pandas": "2.3.3",
    "numpy": "2.4.6",
    "scipy": "1.17.1",
    "joblib": "1.5.3",
    "xgboost-cpu": "3.4.1",
    "catboost": "1.2.10",
}

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*(f"{name}=={version}" for name, version in IMAGE_PIN.items()), "fastapi[standard]")
    # For app.training.core.load_bundle, which rebuilds pipelines from the
    # portable model format. Deploy from Backend/ so `app` is importable.
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

        from app.training.core import load_bundle

        # The backend writes models as /models/<run_id>/model.joblib, and calls
        # vol.commit() so the file is visible here. We also call reload() to be
        # sure our container's view is fresh — cheap on a hot path.
        models_volume.reload()
        path = f"/models/{run_id}/model.joblib"
        bundle = load_bundle(joblib.load(path))

        # Evict oldest if at capacity.
        if len(self._cache) >= self._CACHE_LIMIT:
            self._cache.pop(next(iter(self._cache)))
        self._cache[run_id] = bundle
        return bundle

    @modal.fastapi_endpoint(method="POST", docs=True)
    def predict(self, run_id: str, payload: dict):
        """POST /?run_id=<id> with { "features": {...} } or { "rows": [[...]] }.

        Same request/response contract as the old per-run serve_template.py so
        the FastAPI backend's proxy doesn't need to change.
        """
        import pandas as pd

        try:
            bundle = self._bundle(run_id)
        except FileNotFoundError:
            return {"error": f"Model for run_id={run_id} not found in volume."}
        except ValueError as e:
            return {"error": str(e)}

        model = bundle["model"]
        feature_cols = bundle["feature_cols"]
        problem_type = bundle["problem_type"]
        model_name = bundle.get("model_name", "model")
        class_labels = bundle.get("class_labels")

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
            import numpy as np

            # ravel: CatBoost multiclass predict returns shape (n, 1).
            preds_list = np.asarray(model.predict(df)).ravel().tolist()
            result = {"predictions": preds_list, "model": model_name}

            if problem_type == "classification":
                if class_labels:
                    result["predicted_labels"] = [
                        class_labels[int(p)] if 0 <= int(p) < len(class_labels) else None
                        for p in preds_list
                    ]
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(df)
                    result["probabilities"] = probs.tolist()
                    if class_labels:
                        result["class_labels"] = list(class_labels)
            return result
        except Exception as e:
            return {"error": f"Prediction failed: {e}"}

    @modal.fastapi_endpoint(method="GET", docs=True)
    def schema(self, run_id: str):
        """GET /?run_id=<id> — column names and example payloads."""
        try:
            bundle = self._bundle(run_id)
        except FileNotFoundError:
            return {"error": f"Model for run_id={run_id} not found in volume."}
        except ValueError as e:
            return {"error": str(e)}

        feature_cols = bundle["feature_cols"]
        class_labels = bundle.get("class_labels")
        return {
            "run_id": run_id,
            "model_name": bundle.get("model_name", "model"),
            "problem_type": bundle.get("problem_type"),
            "feature_cols": feature_cols,
            "class_labels": list(class_labels) if class_labels else None,
            "example_features": {
                "features": {c: 0 for c in feature_cols}
            },
            "example_rows": {
                "rows": [[0 for _ in feature_cols]]
            },
        }
