"""Upload a trained model into the shared `modelforge-models` Modal Volume.

There is no per-run Modal app anymore. A single permanent `modelforge-inference`
app (see app/deploy/inference_app.py) serves predictions for every run by
reading models out of the volume on demand. The "deploy" action here just
puts this run's model.joblib in the right spot and stores the public URLs
the frontend should hit.

Designed to be invoked as a FastAPI background task. Writes deployment.json
into the run directory so the frontend can poll status.
"""
import logging
import time
from datetime import datetime, timezone

from app import storage
from app.config import settings

log = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_deployment(run_id: str, **fields) -> None:
    existing = storage.read_json(run_id, "deployment.json") or {}
    existing.update(fields)
    storage.write_json(run_id, "deployment.json", existing)


def _endpoint_url(workspace: str, method: str, run_id: str) -> str:
    """Build a Modal fastapi_endpoint URL for the shared inference app.

    Modal exposes class methods as:
        https://{workspace}--{app}-{class-lower}-{method}.modal.run
    With our Predictor class that's `predictor-predict` and `predictor-schema`.
    """
    app = settings.modal_inference_app
    host = f"{workspace}--{app}-predictor-{method}.modal.run"
    return f"https://{host}/?run_id={run_id}"


def deploy_run(run_id: str) -> dict:
    """Upload model to the shared Modal Volume and persist endpoint URLs.

    Returns the resulting deployment dict (also persisted to deployment.json).
    """
    started = time.time()
    model_path = storage.artifact_path(run_id, "model.joblib")  # restores from R2 if needed

    if not model_path.exists():
        err = "model.joblib missing — has the training run succeeded?"
        _write_deployment(run_id, status="failed", error=err, finished_at=_now())
        return {"status": "failed", "error": err}

    if not settings.modal_workspace:
        err = (
            "MODAL_WORKSPACE is not configured. Set it in Backend/.env to your "
            "Modal workspace username (visible in `modal app list` output)."
        )
        log.error(err)
        _write_deployment(run_id, status="failed", error=err, finished_at=_now())
        return {"status": "failed", "error": err}

    app_name = settings.modal_inference_app
    _write_deployment(
        run_id,
        status="deploying",
        app_name=app_name,
        started_at=_now(),
        predict_url=None,
        schema_url=None,
        error=None,
    )

    try:
        import modal
    except ImportError:
        err = (
            "modal package not importable from the backend venv. "
            "Run `pip install modal` in the same venv that runs uvicorn."
        )
        log.exception(err)
        _write_deployment(run_id, status="failed", error=err, finished_at=_now())
        return {"status": "failed", "error": err}

    # Open (or create on first use) the shared models volume and drop this
    # run's bundle into it. batch_upload commits automatically when the
    # context manager exits, so the inference container will see the new
    # file on its next request.
    try:
        log.info("Uploading model for run %s to volume %s", run_id, settings.modal_models_volume)
        vol = modal.Volume.from_name(
            settings.modal_models_volume,
            create_if_missing=True,
        )
        remote_path = f"/{run_id}/model.joblib"
        with vol.batch_upload(force=True) as batch:
            batch.put_file(str(model_path), remote_path)
    except Exception as e:
        err = f"Failed to upload model to Modal Volume: {e}"
        log.exception(err)
        _write_deployment(run_id, status="failed", error=err, finished_at=_now())
        return {"status": "failed", "error": err}

    predict_url = _endpoint_url(settings.modal_workspace, "predict", run_id)
    schema_url = _endpoint_url(settings.modal_workspace, "schema", run_id)
    elapsed = round(time.time() - started, 2)

    deployment = {
        "status": "succeeded",
        "app_name": app_name,
        "predict_url": predict_url,
        "schema_url": schema_url,
        "all_urls": [predict_url, schema_url],
        "elapsed_seconds": elapsed,
        "finished_at": _now(),
        "error": None,
    }
    _write_deployment(run_id, **deployment)
    log.info("Model for run %s uploaded in %.2fs => %s", run_id, elapsed, predict_url)
    return deployment
