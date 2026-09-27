import json as _json
import logging
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import joblib
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from app import jobs, r2, storage
from app.config import settings
from app.data.images import ingest_image_zip
from app.data.ingest import SUPPORTED_EXTENSIONS, IngestError, ingest_file
from app.deploy.modal_deploy import deploy_run
from app.pipeline import export, tasks
from app.pipeline.train import GPU_APP_NAME, TRAINING_GPU
from app.schemas import DirectUploadRequest, StartRunRequest

logging.basicConfig(level=settings.log_level)
log = logging.getLogger("modelforge")

app = FastAPI(title="Namtheg Backend (MVP)", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_EXTENSIONS = (*SUPPORTED_EXTENSIONS, ".zip")
MAX_UPLOAD_BYTES = 30 * 1024 * 1024  # through the frontend proxy
MAX_DIRECT_UPLOAD_BYTES = 2 * 1024 ** 3  # browser straight to R2


@app.middleware("http")
async def fresh_run_state(request: Request, call_next):
    """Runs execute in their own job containers; reload the shared volume so
    polls see their latest status and artifacts."""
    if request.method == "GET" and request.url.path.startswith("/runs/"):
        await run_in_threadpool(storage.sync_reload)
    return await call_next(request)


@app.api_route("/health", methods=["GET", "HEAD"])
def health() -> dict:
    return {
        "status": "ok",
        "llm_configured": bool(settings.openai_api_key or settings.openrouter_api_key),
        "llm_provider": "openai" if settings.openai_api_key else "openrouter",
        "llm_model": settings.llm_primary_model if settings.openai_api_key else settings.openrouter_model,
        "llm_primary": {"model": settings.llm_primary_model, "configured": bool(settings.openai_api_key)},
        "llm_fallback": {"model": settings.openrouter_model, "configured": bool(settings.openrouter_api_key)},
        "sandbox_backend": settings.sandbox_backend,
        "training_service": GPU_APP_NAME,
        "r2_enabled": r2.enabled(),
    }


def _ingest(run_id: str, raw: Path, filename: str) -> dict:
    """Parse a stored upload into the run's canonical dataset (blocking)."""
    if raw.suffix.lower() == ".zip":
        report = ingest_image_zip(raw, storage.run_dir(run_id))
    else:
        report = ingest_file(raw, storage.run_dir(run_id) / storage.DATASET_FILE, filename)
    storage.persist(run_id, f"raw/{raw.name}")
    storage.persist(run_id, storage.DATASET_FILE)
    storage.write_json(run_id, "ingest.json", report)
    storage.write_status(run_id, "uploaded", filename=filename, source_format=report["source_format"])
    return report


def _upload_response(run_id: str, filename: str, report: dict) -> dict:
    df = storage.dataset_head(run_id, 5)
    # Roundtrip through pandas' JSON writer to safely handle NaN, Inf,
    # Timestamps and numpy scalar types that the default encoder rejects.
    return {
        "run_id": run_id,
        "filename": filename,
        "columns": [str(c) for c in df.columns],
        "preview": _json.loads(df.to_json(orient="records", date_format="iso")),
        "n_rows": report["n_rows"],
        "ingest": {k: report.get(k) for k in ("source_format", "actions", "warnings", "classes")},
    }


def _check_extension(filename: str | None) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in UPLOAD_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Accepted: {', '.join(UPLOAD_EXTENSIONS)}.")
    return ext


@app.post("/upload")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    """Store the raw upload, convert it once to the run's dataset (a typed
    Parquet DataFrame, or an image manifest for zips), and return a preview.
    The frontend always prefers /uploads/direct (straight to R2, any size);
    this is only the fallback for when the backend has no R2 configured, so
    it's capped at 30 MB."""
    ext = _check_extension(file.filename)
    run_id = storage.new_run_id()
    raw = storage.raw_upload_path(run_id, ext)
    total_bytes = 0
    try:
        with raw.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "File too large (30 MB cap without R2 configured). Set R2_* in "
                                             "Backend/.env to accept uploads of any size.")
                f.write(chunk)
        report = await run_in_threadpool(_ingest, run_id, raw, file.filename)
    except HTTPException:
        shutil.rmtree(storage.run_dir(run_id), ignore_errors=True)
        raise
    except IngestError as e:
        shutil.rmtree(storage.run_dir(run_id), ignore_errors=True)
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("Upload failed for run %s", run_id)
        shutil.rmtree(storage.run_dir(run_id), ignore_errors=True)
        raise HTTPException(500, f"Upload failed: {e}")
    return _upload_response(run_id, file.filename, report)


@app.post("/uploads/direct")
def start_direct_upload(req: DirectUploadRequest) -> dict:
    """The primary upload path, whatever the file's size: returns a
    time-limited URL the browser PUTs the file straight to in R2, skipping the
    backend and the frontend proxy entirely; /uploads/{run_id}/complete then
    ingests it. 409 when this backend has no R2 configured — the frontend
    falls back to the (30 MB-capped) /upload proxy in that case."""
    if not r2.enabled():
        raise HTTPException(409, "Direct uploads need R2 storage, which is not configured on this backend.")
    ext = _check_extension(req.filename)
    if req.size > MAX_DIRECT_UPLOAD_BYTES:
        raise HTTPException(413, "File too large. Maximum size is 2 GB.")
    run_id = storage.new_run_id()
    storage.write_status(run_id, "uploading", filename=req.filename, size=req.size)
    return {"run_id": run_id, "method": "PUT",
            "upload_url": r2.presigned_upload_url(run_id, f"raw/source{ext}"),
            "expires_in": settings.r2_url_ttl_seconds}


@app.post("/uploads/{run_id}/complete")
async def complete_direct_upload(run_id: str) -> dict:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    status_data = storage.read_status(run_id)
    if status_data.get("status") != "uploading":
        raise HTTPException(409, "This upload was already completed.")
    ext = Path(status_data["filename"]).suffix.lower()
    raw = storage.raw_upload_path(run_id, ext)
    if not r2.download(run_id, f"raw/source{ext}", raw):
        raise HTTPException(409, "The file hasn't arrived in storage yet.")
    if raw.stat().st_size > MAX_DIRECT_UPLOAD_BYTES:
        raise HTTPException(413, "File too large. Maximum size is 2 GB.")
    try:
        report = await run_in_threadpool(_ingest, run_id, raw, status_data["filename"])
    except IngestError as e:
        storage.write_status(run_id, "failed", error=str(e))
        raise HTTPException(400, str(e))
    return _upload_response(run_id, status_data["filename"], report)


@app.get("/runs/{run_id}/preview")
def preview(run_id: str) -> dict:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    df = storage.dataset_head(run_id, 20)
    status_data = storage.read_status(run_id) or {}
    filename = status_data.get("filename", "dataset.csv")
    columns = [str(c) for c in df.columns.tolist()]
    preview_rows = _json.loads(df.to_json(orient="records", date_format="iso"))
    return {
        "run_id": run_id,
        "filename": filename,
        "source_format": status_data.get("source_format"),
        "n_columns": int(df.shape[1]),
        "columns": columns,
        "preview": preview_rows,
    }


@app.post("/runs/{run_id}/start")
def start_run(run_id: str, req: StartRunRequest, background: BackgroundTasks) -> dict:
    """Validate the task, then start the run in the background (a Modal job in
    production). Poll /runs/{run_id}/status for progress."""
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    if storage.read_status(run_id).get("status") in ("queued", "running"):
        raise HTTPException(409, "This run is already in progress.")
    try:
        spec = tasks.resolve(run_id, req.task, req.target, req.date_column, req.series_id_column, req.horizon)
    except tasks.TaskError as e:
        raise HTTPException(400, str(e))
    storage.write_status(run_id, "queued", task=spec["task"], target=spec.get("target"), queued_at=time.time())
    storage.update_status(run_id, **jobs.start(run_id, spec, background))
    return {"run_id": run_id, "status": "queued", **spec}


@app.post("/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    status_data = storage.read_status(run_id)
    if status_data.get("status") not in ("queued", "running"):
        raise HTTPException(409, f"The run is {status_data.get('status')}; nothing to cancel.")
    if not jobs.cancel(status_data):
        raise HTTPException(409, "Runs executing in the local development server can't be cancelled.")
    storage.write_status(run_id, "cancelled", error="Cancelled by the user.")
    return {"run_id": run_id, "status": "cancelled"}


@app.get("/runs/{run_id}/status")
def status(run_id: str) -> dict:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    return {"run_id": run_id, **storage.read_status(run_id)}


@app.get("/runs/{run_id}/result")
def result(run_id: str) -> JSONResponse:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    result = storage.read_json(run_id, "result.json")
    if result is None:
        raise HTTPException(409, "Run has not produced a result yet.")
    return JSONResponse(result)


DOWNLOADS = {
    # kind: (artifact, filename suffix)
    "cleaned_csv": (export.CLEANED_CSV, "_cleaned.csv"),
    "model": (export.MODEL_PACKAGE, "_model.zip"),
    "forecast": (export.FORECAST_CSV, "_forecast.csv"),
}


@app.get("/runs/{run_id}/download/{kind}")
def download(run_id: str, kind: str):
    """cleaned_csv: the dataset the models trained on. model: the winning model
    package (weights, runtime, metadata, predict.py). forecast: the forecast
    with 80% intervals. With R2 on, this redirects to a presigned URL that
    expires after R2_URL_TTL_SECONDS."""
    if kind not in DOWNLOADS:
        raise HTTPException(404, f"Unknown download {kind!r}. Available: {', '.join(DOWNLOADS)}.")
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    name, suffix = DOWNLOADS[kind]
    stem = Path((storage.read_status(run_id) or {}).get("filename") or "dataset").stem
    filename = f"{stem}{suffix}"
    if r2.exists(run_id, name):
        return RedirectResponse(r2.presigned_download_url(run_id, name, filename), status_code=307)
    path = storage.artifact_path(run_id, name)
    if not path.exists():
        raise HTTPException(409, "Not available for this run.")
    return FileResponse(path, filename=filename)


@app.get("/runs/{run_id}/agent_log")
def agent_log(run_id: str) -> dict:
    """Every code cell the analyst agent ran in the sandbox, with its output."""
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    data = storage.read_json(run_id, "agent_log.json")
    if data is None:
        raise HTTPException(409, "The analysis has not run yet.")
    return {"run_id": run_id, **data}


@app.get("/runs/{run_id}/plot")
def plot(run_id: str) -> FileResponse:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    path = storage.artifact_path(run_id, "plot.png")
    if not path.exists():
        raise HTTPException(404, "Plot not yet generated.")
    return FileResponse(path, media_type="image/png", filename="plot.png")


@app.post("/runs/{run_id}/deploy")
def deploy(run_id: str, background: BackgroundTasks) -> dict:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    if not storage.artifact_path(run_id, "model.joblib").exists():
        raise HTTPException(409, "No trained model to deploy - finish a successful run first.")

    existing = storage.read_json(run_id, "deployment.json") or {}
    if existing.get("status") == "deploying":
        return {"run_id": run_id, "status": "deploying", "message": "Deployment already in progress."}

    storage.write_json(
        run_id,
        "deployment.json",
        {"status": "deploying", "app_name": settings.modal_inference_app},
    )
    background.add_task(deploy_run, run_id)
    return {"run_id": run_id, "status": "deploying"}


@app.get("/runs/{run_id}/deployment")
def deployment(run_id: str) -> dict:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    data = storage.read_json(run_id, "deployment.json")
    if data is None:
        return {"run_id": run_id, "status": "not_deployed"}
    return {"run_id": run_id, **data}


@app.get("/runs/{run_id}/model_schema")
def model_schema(run_id: str) -> dict:
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    # model_meta.json carries everything this endpoint needs, so the backend
    # never unpickles a model. Runs from before the GPU trainer only have the
    # sklearn pickle.
    bundle = storage.read_json(run_id, "model_meta.json")
    if bundle is None:
        bundle_path = storage.artifact_path(run_id, "model.joblib")
        if not bundle_path.exists():
            raise HTTPException(409, "No trained model - finish a successful run first.")
        bundle = joblib.load(bundle_path)

    # First engineered row gives a realistic baseline for the predict form.
    sample: dict = {}
    eng_path = storage.engineered_path(run_id)
    if bundle["feature_cols"] and (eng_path.exists() or (storage.run_dir(run_id) / "engineered.csv").exists()):
        row = storage.engineered_head(run_id, 1)
        target = storage.read_status(run_id).get("target")
        for c in bundle["feature_cols"]:
            if c in row.columns:
                val = row.iloc[0][c]
                # Cast numpy scalars to plain JSON-safe types.
                if pd.isna(val):
                    sample[c] = None
                elif hasattr(val, "item"):
                    sample[c] = val.item()
                else:
                    sample[c] = val
        # paranoia: never leak the target column
        sample.pop(target, None)

    return {
        "run_id": run_id,
        "task": bundle.get("task") or bundle.get("problem_type"),
        "model_name": bundle.get("model_name"),
        "problem_type": bundle.get("problem_type"),
        "feature_cols": bundle["feature_cols"],
        "class_labels": bundle.get("class_labels"),
        "forecast": bundle.get("forecast"),
        "sample": sample,
    }


@app.post("/runs/{run_id}/predict")
async def predict(run_id: str, request: Request) -> dict:
    """Proxy a prediction request to the deployed Modal endpoint.

    Going through the backend avoids browser CORS issues with Modal."""
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    dep = storage.read_json(run_id, "deployment.json") or {}
    predict_url = dep.get("predict_url")
    if not predict_url or dep.get("status") != "succeeded":
        raise HTTPException(409, "Model is not deployed yet. Click 'Deploy to Modal' first.")

    payload = await request.body()
    req = urllib.request.Request(
        predict_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")
        raise HTTPException(e.code, f"Modal endpoint error: {detail[:500]}")
    except urllib.error.URLError as e:
        raise HTTPException(502, f"Could not reach Modal endpoint: {e.reason}")
    try:
        return _json.loads(body)
    except _json.JSONDecodeError:
        raise HTTPException(502, f"Modal returned non-JSON: {body[:500]}")


@app.get("/runs/{run_id}/diagnostics")
def get_run_diagnostics(run_id: str) -> dict:
    """Where the run is and what it's running on, from recorded state only.
    Training happens on separate GPU containers, so this API container's own
    CPU and memory say nothing about it and are not reported."""
    if not storage.run_exists(run_id):
        raise HTTPException(404, "run_id not found")
    s = storage.read_status(run_id)
    metrics = storage.read_json(run_id, "metrics.json") or {}
    hardware = (metrics.get("extra") or {}).get("hardware") or {}
    queued_at = s.get("queued_at")
    return {
        "status": s.get("status"),
        "stage": s.get("stage"),
        "task": s.get("task"),
        "gpu": hardware.get("gpu") or f"NVIDIA {TRAINING_GPU} (requested)",
        "gpu_calls": len(s.get("gpu_calls") or []),
        "elapsed_seconds": round(time.time() - queued_at) if queued_at else None,
    }
